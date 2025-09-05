from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Set, Tuple, Optional
from nicegui import ui, app
from botifarra.botifarra_env_v2 import BotifarraEnvV2
from botifarra.rl_utils import code_card, decode_action_card, one_hot_encode_hand
from botifarra.pals import BOTIFARRA
from botifarra.dqn_botifarra import DQNBotifarra
import numpy as np

import time


# --------------------------- ASSETS ---------------------------------
# Serveix els fitxers estàtics (cartes 0..47.png) des de ./static
app.add_static_files('/static', 'static')  # espera ./static/cards/NN.png

# --------------------------- MODEL ----------------------------------

SEAT_NAME = {0: 'J1', 1: 'J2', 2: 'J3', 3: 'J4'}


class Game:
    def __init__(self, room: str, players_mode: Dict[int, str] = {0: 'huma', 1: 'IA', 2: 'IA', 3: 'IA'}):
        self.room = room
        self.players = {s: [] for s in range(4)}
        self.table = []  # (seat, card)
        self.team_points = {'A': 0, 'B': 0}
        self.total_points = {'A': 0, 'B': 0}
        self.turn = -1  # jugador actiu (0...3), -1 ningú (estem cantant!)
        self.canta = 1  # jugador que canta (0..3)
        self.cantant = True  # estem en fase de cantar trumfo
        self.delegats = False  # si el jugador que canta ha delegat
        self.log = []
        self.subscribers = set()  # redibuix de cada vista
        self.joc = BotifarraEnvV2()
        self.agent_IA = DQNBotifarra()
        self.last_obs = None
        self.last_mask = None
        self.wait = 0
        self.punts_partida = 50
        self.last_redraw = time.time()
        self.PLAYER_MODE = players_mode

        # Load IA weights
        self.agent_IA.load_weights("../agents/botifarra_v2_200k_dqn")

    def broadcast(self):
        """Re-pinta totes les vistes d'aquesta partida."""
        for fn in list(self.subscribers):
            try:
                fn()
            except Exception:
                # si una vista ha marxat, la treiem
                self.subscribers.discard(fn)

    # ---------- joc ----------
    def set_trumfo(self, t_id):
        # el jugador humà ha delegat
        if t_id == 5:
            # delega ....
            self.delegats = True
            self.log.append(f'El jugador {SEAT_NAME[self.canta]} delega...')
            self.canta = (self.canta + 2) % 4
        # El jugador humà a cantat trumfo
        else:
            self.joc.trumfo = t_id
            self.log.append(f'El jugador {SEAT_NAME[self.canta]} canta {self.joc.pals[self.joc.trumfo]}')
            
            if self.delegats:
                self.joc.jugador_actual = (self.canta - 1) % 4  # el jugador
                self.canta = (self.canta - 1) % 4  # pòxim jugador a cantar
            else:
                self.joc.jugador_actual = (self.canta + 1) % 4
                self.canta = (self.canta + 1) % 4  # pòxim jugador a cantar

            # Agafa observació i mask per agent_IA
            self.last_obs = self.joc.get_state(self.joc.jugador_actual)
            self.last_mask = one_hot_encode_hand(
                self.joc.jugadors[self.joc.jugador_actual].cartes_valides(self.joc.trumfo, self.joc.taula)[0]
            )

            self.cantant = False
            self.broadcast()

    # Juguem nova mà
    def new_deal(self):
        # Repartir cartes
        self.joc.reset_joc()
        self.joc.repartir_cartes()

        for s in range(4):
            self.players[s].clear()
            for c in self.joc.jugadors[s].ma:
                self.players[s].append(code_card(c))

        self.table.clear()
        self.log.append(f'Nova mà repartida canta el jugador {SEAT_NAME[self.canta]}')
        self.broadcast()

        # Cantar trumfo
        self.cantant = True
        self.delegats = False

    # Juguem una carta
    def play_card(self, seat: int, card: int):
        if seat != self.joc.jugador_actual:
            return
        if card not in self.players[seat]:
            return
        if len(self.table) >= 4:
            return

        # Mira cartes vàlides pel jugador
        ma_valida = self.joc.jugadors[seat].cartes_valides(self.joc.trumfo, self.joc.taula)[0]
        carta_jugada = decode_action_card(card)
        if carta_jugada not in ma_valida:
            self.log.append(f'{SEAT_NAME[seat]}, no pots jugar aquesta carta!')
            self.broadcast()
            return

        # Actualitza l'estat del joc
        self.joc.taula.append(carta_jugada)
        # treu la carta de la mà del jugador
        self.joc.jugadors[seat].ma.remove(carta_jugada)
        # Actualitzem l'històric de jugades amb la carta jugada
        self.joc.update_state(seat, card, None)

        # Eliminem carta de la visualització i de l'engine i l'afegim a la taula
        self.players[seat].remove(card)
        self.table.append((seat, card))
        self.log.append(f'{SEAT_NAME[seat]} juga {decode_action_card(card)}')

        # quan hi ha 4 cartes a taula, simulem "tancar la basa"
        if len(self.table) < 4:
            # passa el torn al següent
            self.joc.jugador_actual = (self.joc.jugador_actual + 1) % 4
            
            # Agafa observació i mask per agent_IA
            self.last_obs = self.joc.get_state(self.joc.jugador_actual)
            ma_valida = self.joc.jugadors[self.joc.jugador_actual].cartes_valides(self.joc.trumfo, self.joc.taula)[0]
            self.last_mask = one_hot_encode_hand(ma_valida)

        self.broadcast()

# --------------------------- LÒGICA DE SALES ------------------------

rooms: Dict[str, Game] = {}

def get_game(room: str) -> Game:
    if room not in rooms:
        rooms[room] = Game(room=room)
        rooms[room].new_deal()
    return rooms[room]

# --------------------------- UTILITATS UI ---------------------------

CARD_W_TABLE = 90   # px
CARD_W_HAND = 70    # px

def card_img_src(card_id: int) -> str:
    return f'/static/cards/{card_id}.png'

def player_box(name: str, active: bool):
    with ui.card().classes(
        'w-20 h-12 items-center justify-center flex border '
        + ('border-blue-600 bg-blue-400' if active else 'border-black')
    ):
        ui.label(name).classes('text-lg')


# --------------------------- VISTA PER JUGADOR ----------------------
@ui.page('/game/{room}/{seat}')
def game_page(room: str, seat: int):
    assert seat in range(4), 'Seat must be one of 1..4'
    g = get_game(room)

     # Contenidor global centrat i amb amplada màxima
    with ui.column().classes('w-full items-center'):
        with ui.column().classes('w-full max-w-[980px] mx-auto items-center gap-4'):
            # ---------- capçalera ----------
            title = ui.label().classes('text-2xl font-bold')

            columns = [
                {'name': 'equip', 'label': '', 'field': 'equip', 'align': 'left', 'headerClasses': 'font-bold'},
                {'name': 'A', 'label': 'A', 'field': 'A', 'align': 'center', 'headerClasses': 'font-bold'},
                {'name': 'B', 'label': 'B', 'field': 'B', 'align': 'center', 'headerClasses': 'font-bold'},
            ]

            # fila amb taula i trumfo al costat
            with ui.row().classes('gap-8 items-start my-4'):
                score_table = ui.table(columns=columns, rows=[]).classes(
                    'text-center border-collapse border border-gray-400 text-xl'
                ).style('width:250px;')

                # trumfo: element fix, només canvia la source
                trumfo_card = ui.image().props('no-transition no-spinner').style('width:95px;')

            # fila pels jugadors (la deixo tal qual)
            with ui.row().classes('gap-4 my-2'):
                top_row = ui.row().classes('gap-4')  # caixetes jugadors

            action_log = ui.label().classes('italic text-slate-600 my-2 text-lg')

            # ---------- CANTA: slots fixos (handlers fixos) ----------
            CANTA_MAX = 6  # màxim (incloent delegats)
            canta_row = ui.row().classes('gap-4 my-6')
            canta_imgs: list = []
            canta_choice: list = [None] * CANTA_MAX  # valor viu per slot

            with canta_row:
                for i in range(CANTA_MAX):
                    im = ui.image().props('no-transition no-spinner').style(f'width:{int(CARD_W_TABLE*4/6)}px; display:none;')

                    # handler fix que consulta el valor del slot en el moment del clic
                    def make_c_handler(idx, im_ref=im):
                        def _on_click(e):
                            t_id = canta_choice[idx]
                            if t_id is not None:
                                # antirebot (opcional)
                                im_ref.props('disable')
                                try:
                                    g.set_trumfo(t_id)
                                finally:
                                    im_ref.props(remove='disable')
                        return _on_click

                    im.on('click', make_c_handler(i))
                    canta_imgs.append(im)

            # ---------- TAULA: 4 slots fixos (sense clear) ----------
            TABLE_SLOTS = 4
            table_row = ui.row().classes('gap-4 my-6')
            table_imgs, table_labels = [], []
            with table_row:
                for _ in range(TABLE_SLOTS):
                    with ui.column().classes('items-center'):
                        im = ui.image().props('no-transition no-spinner').style(f'width:{CARD_W_TABLE}px; display:none;')
                        lb = ui.label('').classes('text-sm').style('display:none;')
                        table_imgs.append(im)
                        table_labels.append(lb)

            # ---------- MÀ: 12 slots fixos (handlers fixos) ----------
            HAND_MAX = 12
            hand_row = ui.row().classes('gap-2 my-4')
            # ui.label('La teva mà').classes('mr-2 font-semibold')

            hand_imgs: list = []
            hand_slot_card: list = [None] * HAND_MAX   # valor viu de cada slot

            with hand_row:
                for i in range(HAND_MAX):
                    im = ui.image().props('no-transition no-spinner').classes(
                        'cursor-pointer hover:scale-105 transition'
                    ).style(f'width:{CARD_W_HAND}px; display:none;')

                    # Handler FIX per slot: sempre consulta hand_slot_card[i]
                    def make_handler(idx, im_ref=im):
                        def _on_click(e):
                            card_id = hand_slot_card[idx]
                            if card_id is not None:
                                im_ref.props('disable')
                                try:
                                    g.play_card(seat, card_id)
                                finally:
                                    im_ref.props(remove='disable')
                        return _on_click

                    im.on('click', make_handler(i))
                    hand_imgs.append(im)

            # ---------- estat previ per evitar updates innecessaris ----------
            prev = {
                'trumfo_img': None,
                'table': [],
                'hand': [],
                'canta_on': None,
                'scoreA': None, 'scoreB': None, 'totalA': None, 'totalB': None,
                'title': None, 'log_tail': None,
                'turn': None,
            }

    # ---------- helpers ----------
    def show_img(img, src):
        if img is None:
            return
        img.set_source(src)
        img.style('display:block;')

    def hide_img(img):
        if img is None:
            return
        img.style('display:none;')

    # ---------- REDRAW sense flicker ----------
    def redraw():
        g.last_redraw = time.time()

        # Títol (només si canvia)
        new_title = f'Partida {g.room} — Jugador {SEAT_NAME[seat]} ({g.PLAYER_MODE[seat]})'
        if prev['title'] != new_title:
            title.set_text(new_title)
            prev['title'] = new_title

        # Sempre actualitza (per evitar errors de concurrència)
        score_table.rows = [
            {'equip': 'TOTAL:',   'A': g.total_points['A'], 'B': g.total_points['B']},
            {'equip': 'PARCIAL:', 'A': g.team_points['A'],  'B': g.team_points['B']},
        ]
        score_table.update()

        # Jugadors actiu (deixes el teu clear; si vols, també es pot fer in-place)
        top_row.clear()
        with top_row:
            for s in range(4):
                player_box(SEAT_NAME[s], active=(g.joc.jugador_actual == s))

        # Trumfo (només canvia la source)
        img_trumfo = g.joc.trumfo * 12 if g.joc.trumfo != -1 else 50
        if g.joc.trumfo == 4:
            img_trumfo = 48
        if prev['trumfo_img'] != img_trumfo:
            trumfo_card.set_source(card_img_src(img_trumfo))
            prev['trumfo_img'] = img_trumfo

        # CANTA (slots fixos + array de valors)
        want_canta = (g.canta == seat and g.cantant)
        if prev['canta_on'] != want_canta:
            # reset visuals
            for i in range(CANTA_MAX):
                canta_choice[i] = None
                hide_img(canta_imgs[i])

            if want_canta:
                v = 6 if not g.delegats else 5
                for i, c in enumerate(range(v)):
                    canta_choice[i] = c
                    show_img(canta_imgs[i], card_img_src(c * 12))
            prev['canta_on'] = want_canta

        # TAULA (sempre 4 imatges: cartes reals + dorsos com a placeholders)
        if prev['table'] != g.table:
            # pinta 0..len(g.table)-1 amb cartes reals i resta amb dors (50.png)
            n = len(g.table)
            for i in range(TABLE_SLOTS):
                if i < n:
                    s, cid = g.table[i]
                    show_img(table_imgs[i], card_img_src(cid))
                    table_labels[i].set_text(SEAT_NAME[s])
                    table_labels[i].style('display:block;')
                    table_imgs[i].classes(remove='opacity-40')  # per si vols diferenciar visualment
                else:
                    show_img(table_imgs[i], card_img_src(50))   # dors de carta
                    table_labels[i].set_text('...')
                    table_imgs[i].classes()     # (opcional) més suau

            prev['table'] = list(g.table)

        # MÀ (slots fixos; NO re-enllaça handlers)
        if prev['hand'] != g.players[seat]:
            current_hand = g.players[seat]

            # amaga tots els slots per defecte i esborra valor
            for i in range(HAND_MAX):
                hand_slot_card[i] = None
                hand_imgs[i].style('display:none;')

            # mostra els necessaris amb el valor actual del slot
            for i, cid in enumerate(current_hand[:HAND_MAX]):
                hand_slot_card[i] = cid
                hand_imgs[i].set_source(card_img_src(cid))
                hand_imgs[i].style('display:inline-block;')

            prev['hand'] = list(g.players[seat])

        # Log curt i torn
        log_tail = ''.join(g.log[-1:]) + (f'   | Torn: {SEAT_NAME[g.joc.jugador_actual]}' if not g.cantant else '')
        if prev['log_tail'] != log_tail:
            action_log.set_text(log_tail)
            action_log.style('background-color: #d0ebff; width: 100%; text-align: center; padding: 8px;')
            prev['log_tail'] = log_tail

    # Subscriu aquesta vista i des-subscriu al desconnectar del client
    g.subscribers.add(redraw)
    ui.context.client.on_disconnect(lambda: g.subscribers.discard(redraw))

    # dibuix inicial
    redraw()

    # ---------------- Timer per client CONTROLA TOTA LA LÒGICA DEL JOC I LES IA----------------
    def client_timer_callback():
        if seat != 0:
            redraw()
            return  # només el client 0 controla la lògica del joc
        
        # ------------------ CANTEM EL TRUMFO ------------------
        if g.cantant and (g.PLAYER_MODE[g.canta] == 'IA'): # canta IA
             # Wait 1 cilcle 
            if g.wait == 0:
                g.wait = 1
                redraw()
                return
            g.wait = 0

            g.joc.trumfo = g.joc.jugadors[g.canta].cantar(delegat=g.delegats)
            if g.joc.trumfo >= 0:
                g.log.append(f'El jugador {SEAT_NAME[g.canta]} canta {g.joc.pals[g.joc.trumfo]}')
                g.cantant = False
                if g.delegats:
                    g.joc.jugador_actual = (g.canta - 1) % 4  # el jugador
                    g.canta = (g.canta - 1) % 4  # pòxim jugador a cantar
                else:
                    g.joc.jugador_actual = (g.canta + 1) % 4
                    g.canta = (g.canta + 1) % 4  # pòxim jugador a cantar
                    
                # Preparem per 1a jugada
                # Agafa observació i mask per agent_IA
                g.last_obs = g.joc.get_state(g.joc.jugador_actual)
                g.last_mask = one_hot_encode_hand(g.joc.jugadors[g.joc.jugador_actual].cartes_valides(g.joc.trumfo, g.joc.taula)[0])

            else: # la IA ha delegat
                g.delegats = True
                g.log.append(f'El jugador {SEAT_NAME[g.canta]} delega...')
                g.canta = (g.canta + 2) % 4
        
        if g.cantant and (g.PLAYER_MODE[g.canta] == 'huma'): # canta Huma
            redraw() # Si el jugador humà ha de cantar (directe o delegat) no sempre es feia el redraw bé
            return

        # ------------------ JUGAR CARTA LA IA ------------------  
        if (not g.cantant) and (g.PLAYER_MODE[g.joc.jugador_actual] == 'IA'):
            card = g.agent_IA.choose_action(np.array(g.last_obs), np.array(g.last_mask), deterministic=True)
            g.play_card(g.joc.jugador_actual, card)

        # ------------------ COMPROVEM JUANYADOR DE BASE ------------------
        if len(g.table) == 4:
            # Wait 1 cilcle 
            if g.wait <= 2:
                g.wait += 1
                redraw()
                return
            g.wait = 0
            # g.joc.print_hist()

            idx = g.joc.carta_guanyadora(g.joc.trumfo, g.joc.taula)
            guanyador = g.table[idx][0]
            punts_jugada = sum(carta.get_punts() for carta in g.joc.taula) + 1 # + 1 punt per cada jugada
          
            g.log.append(f'Guanya {SEAT_NAME[guanyador]} i fa {punts_jugada} punts.')
            if guanyador % 2 == 0:
                g.team_points['A'] += punts_jugada
            else:
                g.team_points['B'] += punts_jugada

            g.joc.jugador_actual = guanyador
            g.table.clear()
            g.joc.taula.clear()
            g.joc.jugades_fetes += 1

            # Agafa observació i mask per agent_IA
            g.last_obs = g.joc.get_state(g.joc.jugador_actual)
            g.last_mask = one_hot_encode_hand(g.joc.jugadors[g.joc.jugador_actual].cartes_valides(g.joc.trumfo, g.joc.taula)[0])
            redraw()
        # ------------------ FINAL MA ------------------
        if g.joc.jugades_fetes >= 12:
            if g.team_points['A'] > g.team_points['B']:
                g.log.append(f'Guanya l\'equip A ({g.team_points["A"]} a {g.team_points["B"]})')
                g.total_points['A'] += g.team_points["A"] - 36
            elif g.team_points['B'] > g.team_points['A']:
                g.log.append(f'Guanya l\'equip B ({g.team_points["B"]} a {g.team_points["A"]})')
                g.total_points['B'] += g.team_points["B"] - 36
            else:
                g.log.append(f'Empat a {g.team_points["A"]}')
            g.joc.jugades_fetes = 0

            if g.total_points['A'] >= g.punts_partida:
                g.log.append(f'Partida guanyada per l\'equip A ({g.total_points["A"]} a {g.total_points["B"]})')
                redraw()
            elif g.total_points['B'] >= g.punts_partida:
                g.log.append(f'Partida guanyada per l\'equip B ({g.total_points["B"]} a {g.total_points["A"]})')
                redraw()
            else:
                g.team_points = {'A': 0, 'B': 0}
                redraw()
                g.new_deal()
        # ------------------ REDRAW ------------------
        now = time.time()
        if now - g.last_redraw > 4.0:
            g.last_redraw = now
            redraw()

    ui.timer(2.0, client_timer_callback, active=True)

# --------------------------- ARRANCA -------------------------------

with ui.header().classes('justify-between'):    
    ui.label('🂡 Botifarra — Demo NiceGUI').classes('text-xl font-bold')
    ui.link('J1', '/game/TEST/0')
    ui.link('J2', '/game/TEST/1')
    ui.link('J3', '/game/TEST/2')
    ui.link('J4', '/game/TEST/3')


ui.run(port=8080, reload=False, show=False)  # posa reload=True si vols autorecàrrega
