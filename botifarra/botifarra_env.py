from botifarra.botifarra import Botifarra
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
        self.jugador_inicial = 0  
        self.jugador_actual = 1 # El jugador que comença jugant és el de després del que ha cantat
        self.jugades_fetes = 0
        self.punts_equip_a = 0
        self.punts_equip_b = 0
        self.trumfo = BOTIFARRA
        self.taula = []
        self.historic_mans = np.zeros((4, 48), dtype=int)  # Històric de cartes jugades i pals fallats per cada jugador

    def reset(self):
        self.jugades_fetes = 0
        self.punts_equip_a = 0
        self.punts_equip_b = 0
        self.taula = []
        info = {}
        self.jugador_actual = (self.jugador_inicial + 1) % 4 # El jugador que comença jugant és el de després del que ha cantat
        info['proxim_jugador'] = self.jugador_actual + 1

        # Repartir cartes
        self.repartir_cartes()
        
        # Cantar 
        self.trumfo = self.cantar_trumfo(self.jugador_inicial)
        self.jugador_inicial = (self.jugador_inicial + 1) % 4

        ma_valida, _, _ = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
        info['mask'] = one_hot_encode_hand(ma_valida)  # Màscara d'accions vàlides per al següent estat

        state = self.get_state(self.jugador_actual)
        
        return state, info

    def step(self, action):
        # Aquí es on implementaries la lògica per processar l'acció
        # i avançar el joc. Aquesta és una simplificació.
        reward = 0
        done = False
        terminated = False
        info = {}
        carta_jugada = decode_action_card(action)
        self.taula.append(carta_jugada)
        info['jugada'] = self.jugades_fetes + 1
        info['carta_jugada'] = carta_jugada
        # true la carta de la mà del jugador
        self.jugadors[self.jugador_actual].ma.remove(carta_jugada)

        # Actualitzem l'històric de jugades amb la carta jugada
        self.historic_mans[self.jugador_actual, action] = 1

        if len(self.taula) == 4: # S'ha completat una jugada
            idx_guanyador = self.__carta_guanyadora__(self.trumfo, self.taula)
            guanyador = (self.jugador_actual + 1 + idx_guanyador) % 4
            punts_jugada = sum(carta.get_punts() for carta in self.taula) + 1 # + 1 punt per cada jugada
            info['guanyador'] = guanyador + 1
            info['punts_jugada'] = punts_jugada
            reward = punts_jugada                                         # Si el reward és positu és que ha guanyat l'equip A
            self.taula = []
            self.jugador_actual = guanyador
            self.jugades_fetes += 1
            if self.jugades_fetes == 12:
                done = True
            info['mask'] = one_hot_encode_hand(self.jugadors[self.jugador_actual].ma)  # Màscara d'accions vàlides per la següent mà

        else:
            # ... canviem al següent jugador
            self.jugador_actual = (self.jugador_actual + 1) % 4
        
            ma_valida, falla_pal, falla_trumfo = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
            # ... i actualitzem l'històric de jugades amb els pals/trumfos fallats si n'hi ha
            if falla_pal:
                for i in range(12 * self.taula[0].pal, 12 * self.taula[0].pal + 12):
                    if self.historic_mans[self.jugador_actual, i] == 0:
                        self.historic_mans[self.jugador_actual, i] = -1
            if falla_trumfo and self.trumfo != BOTIFARRA:
                for i in range(12 * self.trumfo, 12 * self.trumfo + 12):
                    if self.historic_mans[self.jugador_actual, i] == 0:
                        self.historic_mans[self.jugador_actual, i] = -1
            info['mask'] = one_hot_encode_hand(ma_valida)  # Màscara d'accions vàlides per al següent estat
    
        info['proxim_jugador'] = self.jugador_actual + 1
        state = self.get_state(self.jugador_actual)
        return state, reward, done, terminated, info

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