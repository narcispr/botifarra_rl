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
PLAYER_MODE = {0: 'huma', 1: 'IA', 2: 'IA', 3: 'IA'}

@dataclass
class Game:
    room: str
    players: Dict[int, List[int]] = field(default_factory=lambda: {s: [] for s in range(4)})
    table: List[Tuple[int, int]] = field(default_factory=list)  # (seat, card)
    team_points: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0})
    turn: int = -1                   # jugador actiu (0...3), -1 ningú (estem cantant!)
    canta: int = 0                   # jugador que canta (0..3)
    cantant: bool = True            # estem en fase de cantar trumfo    
    delegats: bool = False         # si el jugador que canta ha delegat
    log: List[str] = field(default_factory=list)
    subscribers: Set[Callable[[], None]] = field(default_factory=set)  # redibuix de cada vista
    joc: BotifarraEnvV2 = BotifarraEnvV2()
    agent_IA: DQNBotifarra = DQNBotifarra()
    last_obs: Optional[List[float]] = None
    last_mask: Optional[List[int]] = None
    wait: int = 0

    def broadcast(self):
        """Re-pinta totes les vistes d'aquesta partida."""
        for fn in list(self.subscribers):
            try:
                fn()
            except Exception:
                # si una vista ha marxat, la treiem
                self.subscribers.discard(fn)

    # ---------- joc ----------
    def canta_IA(self):
        self.joc.trumfo = self.joc.jugadors[self.canta].cantar()    
        if self.joc.trumfo >= 0:
            self.log.append(f'El jugador {SEAT_NAME[self.canta]} canta {self.joc.pals[self.joc.trumfo]}')
        else:
            self.joc.trumfo = self.joc.jugadors[(self.canta + 2) % 4].cantar(delegat=True)
            self.log.append(f'El jugador {SEAT_NAME[(self.canta + 2) % 4]} canta {self.joc.pals[self.joc.trumfo]} delegats')
            
    def set_trumfo(self, t_id):
        if t_id == 5:
            # delega ....
            self.delegats = True
            self.log.append(f'El jugador {SEAT_NAME[self.canta]} delega...')   
            self.canta = (self.canta + 2) % 4
             
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
            self.last_mask = one_hot_encode_hand(self.joc.jugadors[self.joc.jugador_actual].cartes_valides(self.joc.trumfo, self.joc.taula)[0])

            self.cantant = False
            self.broadcast()
        
    def new_deal(self):
        # Load IA weights
        self.agent_IA.load_weights("/home/narcis/catkin_ws/src/botifarra/agents/botifarra_v2_200k_dqn")
                                   
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
        self.joc.historic_mans[seat, card] = 1
        for i in range(1, 4):
            self.joc.historic_mans[(seat + i) % 4, card] = -1
        
        # Eliminem carta de la visualització i de l'engine i l'afegim a la taula
        self.players[seat].remove(card)
        self.table.append((seat, card))
        self.log.append(f'{SEAT_NAME[seat]} juga {decode_action_card(card)}')
        
        # quan hi ha 4 cartes a taula, simulem "tancar la basa"
        if len(self.table) < 4:
            # passa el torn al següent
            self.joc.jugador_actual = (self.joc.jugador_actual + 1) % 4
        
            ma_valida, falla_pal, falla_trumfo = self.joc.jugadors[self.joc.jugador_actual].cartes_valides(self.joc.trumfo, self.joc.taula)
            # ... i actualitzem l'històric de jugades amb els pals/trumfos fallats si n'hi ha
            if falla_pal:
                for i in range(12 * self.joc.taula[0].pal, 12 * self.joc.taula[0].pal + 12):
                    if self.joc.historic_mans[self.joc.jugador_actual, i] == 0:
                        self.joc.historic_mans[self.joc.jugador_actual, i] = -1
            if falla_trumfo and self.joc.trumfo != BOTIFARRA:
                for i in range(12 * self.joc.trumfo, 12 * self.joc.trumfo + 12):
                    if self.joc.historic_mans[self.joc.jugador_actual, i] == 0:
                        self.joc.historic_mans[self.joc.jugador_actual, i] = -1
            # Agafa observació i mask per agent_IA
            self.last_obs = self.joc.get_state(self.joc.jugador_actual)
            self.last_mask = one_hot_encode_hand(self.joc.jugadors[self.joc.jugador_actual].cartes_valides(self.joc.trumfo, self.joc.taula)[0])

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
        'w-24 h-12 items-center justify-center flex border '
        + ('border-blue-600 bg-blue-50' if active else 'border-black')
    ):
        ui.label(name).classes('text-lg')

# --------------------------- VISTA PER JUGADOR ----------------------

@ui.page('/game/{room}/{seat}')
def game_page(room: str, seat: int):
    assert seat in range(4), 'Seat must be one of 1..4'
    g = get_game(room)
    wait = 0  # per esperar un cicle després de posar la 4a carta
    
    # ---------- elements (referències) ----------
    title = ui.label().classes('text-2xl font-bold')
    score = ui.label().classes('text-lg')

    with ui.row().classes('gap-4 items-center my-2'):
        top_row = ui.row().classes('gap-4')  # jugadors
        ui.row().style('width:20px;')  # Add gap between top_row and trumfo_card
        trumfo_card = ui.image().classes('border-2 border-blue-600 rounded').style('width:80px;')

    canta_row = ui.row().classes('gap-4 my-6')  # cartes a la taula
    table_row = ui.row().classes('gap-4 my-6')  # cartes a la taula
    action_log = ui.label().classes('italic text-slate-600 my-2 text-lg')

    hand_row = ui.row().classes('gap-2 my-4')   # mà privada

    with ui.row().classes('gap-3'):
        ui.button('Nova mà', on_click=lambda: g.new_deal())

    # ---------- funcions d’UI ----------
    def redraw():
        title.set_text(f'Partida {g.room} — Jugador {SEAT_NAME[seat]} ({PLAYER_MODE[seat]})')
        score.set_text(f'Equip A: {g.team_points["A"]}  —  Equip B: {g.team_points["B"]}')

        # Fila jugadors (J1..J4) amb actiu resaltat
        top_row.clear()
        with top_row:
            for s in range(4):
                player_box(SEAT_NAME[s], active=(g.joc.jugador_actual == s))

        # trumfo (pal de partida)
        img_trumfo = g.joc.trumfo * 12 if g.joc.trumfo != -1 else 50
        if g.joc.trumfo == 4:
            img_trumfo = 48
        trumfo_card.set_source(card_img_src(img_trumfo))

        # Canta UI
        canta_row.clear()
        if g.canta == seat:
            with canta_row:
                v = 6 if not g.delegats else 5
                for c in range(v):
                    with ui.column().classes('items-center'):
                        img = ui.image(card_img_src(c * 12)).classes(
                            'cursor-pointer hover:scale-105 transition'
                        ).style('width:80px;')
                        img.on('click', lambda e, t_id=c: g.set_trumfo(t_id))
                        
        # Cartes a la taula
        table_row.clear()
        with table_row:
            for (s, cid) in g.table:
                with ui.column().classes('items-center'):
                    ui.image(card_img_src(cid)).style(f'width:{CARD_W_TABLE}px;')
                    ui.label(SEAT_NAME[s]).classes('text-sm')

        # Mà privada del jugador connectat
        hand_row.clear()
        with hand_row:
            ui.label('La teva mà').classes('mr-2 font-semibold')
            for cid in g.players[seat]:
                img = ui.image(card_img_src(cid)).classes(
                    'cursor-pointer hover:scale-105 transition'
                ).style(f'width:{CARD_W_HAND}px;')
                img.on('click', lambda e, card_id=cid: g.play_card(seat, card_id))

        # Log curt i torn
        action_log.set_text(''.join(g.log[-1:]) + (f'   | Torn: {SEAT_NAME[g.joc.jugador_actual]}' if not g.cantant else ''))

    # Subscriu aquesta vista i des-subscriu al desconnectar del client
    g.subscribers.add(redraw)  # el client ja està connectat en entrar a la pàgina
    ui.context.client.on_disconnect(lambda: g.subscribers.discard(redraw))

    # Dibuix inicial
    redraw()

    # Timer per client
    def client_timer_callback():
        if seat != 0:
            redraw()
            return  # només el client 0 controla la lògica del joc
        
        # ------------------ CANTEM EL TRUMFO ------------------
        if g.cantant and (PLAYER_MODE[g.canta] == 'IA'): # canta IA
             # Wait 1 cilcle 
            if g.wait == 0:
                g.wait = 1
                redraw()
                return
            g.wait = 0

            trumfo = g.joc.jugadors[g.canta].cantar(delegat=g.delegats)
            if g.joc.trumfo >= 0:
                g.log.append(f'El jugador {SEAT_NAME[g.canta]} canta {g.joc.pals[g.joc.trumfo]}')
                g.joc.trumfo = trumfo
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

        # ------------------ JUGAR CARTA LA IA ------------------  
        if (not g.cantant) and (PLAYER_MODE[g.joc.jugador_actual] == 'IA'):
            # if g.wait == 0:
            #     g.wait = 1
            #     redraw()
            #     return
            # g.wait = 0

            # action = g.agent_IA.choose_action(g.last_obs, np.array(g.last_mask), deterministic=True)
            card = g.agent_IA.choose_action(np.array(g.last_obs), np.array(g.last_mask), deterministic=True)
            g.play_card(g.joc.jugador_actual, card)

        # ------------------ COMPROVEM JUANYADOR DE BASE ------------------
        if len(g.table) == 4:
            # Wait 1 cilcle 
            if g.wait == 0:
                g.wait = 1
                redraw()
                return
            g.wait = 0

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

            # Agafa observació i mask per agent_IA
            g.last_obs = g.joc.get_state(g.joc.jugador_actual)
            g.last_mask = one_hot_encode_hand(g.joc.jugadors[g.joc.jugador_actual].cartes_valides(g.joc.trumfo, g.joc.taula)[0])
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
