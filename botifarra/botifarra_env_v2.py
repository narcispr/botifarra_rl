from botifarra.botifarra_env import BotifarraEnv
from botifarra.pals import BOTIFARRA
from botifarra.rl_utils import one_hot_encode_trumfo_v2, one_hot_encode_hand, one_hot_encode_card

from gymnasium import spaces
import numpy as np
from copy import copy 


class BotifarraEnvV2(BotifarraEnv):
    def __init__(self):
        super().__init__()
        # Redefinim l'espai d'observació per compactar més la informació dels companys
        self.observation_space = spaces.Box(low=0, high=1, shape=(240,), dtype=np.int8)
        # Probabilitat de cada carta pels companys. COm que tothom sap les seves, les que no tenim poden 
        # estar a qualsevol de les 3 mans restants, per tant 1/3 = 0.333
        self.prob_mans_companys = np.ones((4, 48), dtype=int) 

    def update_state(self, ultim_jugador: int, action: int, proxim_jugador: int):
        # Actualitzem la probabilitat de tenir la carta jugada a 0 per a tothom
        for i in range(0, 4):
            self.prob_mans_companys[i, action] = 0
        
        # Actualizem estat amb fallos de pal/trumfo si n'hi ha
        _, falla_pal, falla_trumfo = self.jugadors[proxim_jugador].cartes_valides(self.trumfo, self.taula)
        # ... i actualitzem l'històric de jugades amb els pals/trumfos fallats si n'hi ha
        if falla_pal:
            for i in range(12 * self.taula[0].pal, 12 * self.taula[0].pal + 12):
                if self.historic_mans[proxim_jugador, i] == 1:
                    self.historic_mans[proxim_jugador, i] = 0
        if falla_trumfo and self.trumfo != BOTIFARRA:
            for i in range(12 * self.trumfo, 12 * self.trumfo + 12):
                if self.historic_mans[proxim_jugador, i] == 1:
                    self.historic_mans[proxim_jugador, i] = 0

    def get_state(self, id_jugador: int):
        # Retorna l'estat actual del joc com un diccionari o una altra estructura de dades
        trumfo = one_hot_encode_trumfo_v2(self.trumfo)
        ma = one_hot_encode_hand(self.jugadors[id_jugador].ma)
        ma_altres = []
        for c in self.taula:
            ma_altres.extend(one_hot_encode_card(c))
        for i in range(3 - len(self.taula)):
            ma_a = copy(self.prob_mans_companys[((id_jugador + 1) % 4 + i) % 4])
            # fer una mascara  posar a 0 les cartes de ma que valen 1 a self.jugadors[id_jugador].ma
            for carta in self.jugadors[id_jugador].ma:
                ma_a[carta.idx()] = 0
            ma_altres.extend(ma_a.tolist())
        return np.array(trumfo + ma + ma_altres, dtype=np.int8)
        