from botifarra.botifarra import Botifarra
from botifarra.carta import Carta
from botifarra.rl_utils import encode_estat, decode_action_card, one_hot_encode_hand
from botifarra.pals import BOTIFARRA

from gymnasium import Env, spaces
import numpy as np


class BotifarraEnv(Env, Botifarra):
    # Inherits from both Env and Botifarra
    def __init__(self):
        Env.__init__(self)
        Botifarra.__init__(self)

        # Action space is an integer representing the index of the card to play
        self.action_space = spaces.Discrete(48)
        # Observation space can be defined based on the game state representation
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(390,), dtype=np.float32)
        self.historic_mans = np.zeros((4, 48), dtype=int)  # Històric de cartes jugades i pals fallats per cada jugador

    def reset(self):
        # reset partida
        self.reset_partida()
        info = {}
        info['proxim_jugador'] = self.jugador_actual + 1

        # Cantar 
        info['canta'] = self.jugador_inicial + 1
        self.trumfo = self.cantar_trumfo(self.jugador_inicial)
        info['trumfo'] = self.trumfo

        # Omple la màscara d'accions vàlides i el primer estat
        ma_valida, _, _ = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
        info['mask'] = one_hot_encode_hand(ma_valida)  # Màscara d'accions vàlides per al següent estat
        state = self.get_state(self.jugador_actual)
        
        return state, info

    def step(self, action):
        # Aquí es on implementaries la lògica per processar l'acció
        # i avançar el joc. Aquesta és una simplificació.
        info = {}
        reward = 0
        done = False
        terminated = False
        carta_jugada = decode_action_card(action)
        info['jugada'] = self.jugades_fetes + 1
        info['carta_jugada'] = carta_jugada
        
        # Treu la carta de la mà del jugador i la posa a la taula
        self.jugadors[self.jugador_actual].ma.remove(carta_jugada)
        self.taula.append(carta_jugada)
        
        # Guardem últim jugador que ha jugat per actualitzar esatat després
        ultim_jugador = self.jugador_actual

        if len(self.taula) == 4: # S'ha completat una jugada
            # Busquem guanyador de la base
            idx_guanyador = self.carta_guanyadora(self.trumfo, self.taula)
            guanyador = (self.jugador_actual + 1 + idx_guanyador) % 4
            punts_jugada = sum(carta.get_punts() for carta in self.taula) + 1 # + 1 punt per cada jugada
            reward = punts_jugada                                         # Si el reward és positu és que ha guanyat l'equip A
            # Re-iniciem la taula i actualitzem el jugador actual
            self.taula = []
            self.jugador_actual = guanyador
            self.jugades_fetes += 1
            # Omplim informació
            info['guanyador'] = guanyador + 1
            info['punts_jugada'] = punts_jugada
            # info['mask'] = one_hot_encode_hand(self.jugadors[self.jugador_actual].ma)  # Màscara d'accions vàlides per la següent mà
            # Actualitzem puntuació mà
            if guanyador % 2 == 0:
                self.punts_equip_a += punts_jugada
            else:
                self.punts_equip_b += punts_jugada
            # Si última mà, actualitzem puntuació equips i finalitzem episodi
            if self.jugades_fetes == 12:
                # S'ha completat la mà
                if self.punts_equip_a > self.punts_equip_b:
                    info['punts_equip_a'] = self.punts_equip_a - 36
                    info['punts_equip_b'] = 0
                else:
                    info['punts_equip_a'] = 0
                    info['punts_equip_b'] = self.punts_equip_b - 36
                done = True
        else:
            # ... canviem al següent jugador
            self.jugador_actual = (self.jugador_actual + 1) % 4
        
        info['proxim_jugador'] = self.jugador_actual + 1 # depen de mi mitja base o final de base
        
        # Actualitzem informació ma vàlida
        ma_valida, _, _ = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
        info['mask'] = one_hot_encode_hand(ma_valida)  # Màscara d'accions vàlides per al següent estat
        
        # Actualitzem l'estat del joc, la màscara d'accions vàlides i agafem el nou estat
        # IMPORTANT! En cas de voler canviar les observacions només cal sobreescriure aquestes 2 funcions!
        self.update_state(ultim_jugador, action, self.jugador_actual)
        state = self.get_state(self.jugador_actual)

        return state, reward, done, terminated, info

    def update_state(self, ultim_jugador: int, action: int, proxim_jugador: int):
        # Actualitza l'estat del joc després que un jugador hagi jugat una carta
        self.historic_mans[ultim_jugador, action] = 1
        for i in range(1, 4):
            self.historic_mans[(ultim_jugador + i) % 4, action] = -1
        
        # Actualizem estat amb fallos de pal/trumfo si n'hi ha
        _, falla_pal, falla_trumfo = self.jugadors[proxim_jugador].cartes_valides(self.trumfo, self.taula)
        # ... i actualitzem l'històric de jugades amb els pals/trumfos fallats si n'hi ha
        if falla_pal:
            for i in range(12 * self.taula[0].pal, 12 * self.taula[0].pal + 12):
                if self.historic_mans[proxim_jugador, i] == 0:
                    self.historic_mans[proxim_jugador, i] = -1
        if falla_trumfo and self.trumfo != BOTIFARRA:
            for i in range(12 * self.trumfo, 12 * self.trumfo + 12):
                if self.historic_mans[proxim_jugador, i] == 0:
                    self.historic_mans[proxim_jugador, i] = -1

    def get_state(self, id_jugador: int):
        # Retorna l'estat actual del joc com un diccionari o una altra estructura de dades
        state = encode_estat(self.trumfo, self.jugadors[id_jugador].ma, self.taula)

        # Afegir l'històric de jugades del jugador
        state.extend(self.historic_mans[id_jugador].tolist()) # Afegir l'històric de jugades del jugador actual
        state.extend(self.historic_mans[(id_jugador+2)%4].tolist()) # Afegir l'històric de jugades del company del jugador actual
        state.extend(self.historic_mans[(id_jugador+1)%4].tolist()) # Afegir l'històric de jugades del rival dreta del jugador actual
        state.extend(self.historic_mans[(id_jugador+3)%4].tolist()) # Afegir l'històric de jugades del rival esquerra del jugador actual

        # Afegir els punts de l'equip del jugador i del rival normalitzats
        if id_jugador % 2 == 0:
            state.extend([float(self.punts_equip_a/72), float(self.punts_equip_b/72)]) 
        else:
            state.extend([float(self.punts_equip_b/72), float(self.punts_equip_a/72)])

        return np.array(state, dtype=np.float32)