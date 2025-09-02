from botifarra.botifarra_env import BotifarraEnv
from botifarra.pals import BOTIFARRA
from botifarra.rl_utils import one_hot_encode_trumfo_v2, one_hot_encode_hand, one_hot_encode_card, decode_action_card
from botifarra.carta import Carta

from gymnasium import spaces
import numpy as np
from copy import copy 
from typing import Tuple

class BotifarraEnvV2(BotifarraEnv):
    def __init__(self):
        super().__init__()
        # Redefinim l'espai d'observació per compactar més la informació dels companys
        self.observation_space = spaces.Box(low=0, high=1, shape=(240,), dtype=np.int8)
        # Probabilitat de cada carta pels companys. COm que tothom sap les seves, les que no tenim poden 
        # estar a qualsevol de les 3 mans restants, per tant 1/3 = 0.333
        self.prob_mans_companys = np.ones((4, 48), dtype=int) 

    def __basa_del_company__(self) -> bool:
        # Retorna True si la ma del company es coneguda (totes les cartes tenen probabilitat 0 o 1)
        if len(self.taula) == 2 and self.taula[0].get_valor(self.taula[0].pal, self.trumfo) > self.taula[1].get_valor(self.taula[0].pal, self.trumfo):
            return True
        if len(self.taula) == 3 and (self.taula[1].get_valor(self.taula[0].pal, self.trumfo) > self.taula[0].get_valor(self.taula[0].pal, self.trumfo)) and (self.taula[1].get_valor(self.taula[0].pal, self.trumfo) > self.taula[2].get_valor(self.taula[0].pal, self.trumfo)):
            return True  
        return False
    
    def __res_mes_petit_del_pal_que__(self, jugador: int, carta: Carta):
        ordre = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 1, 9] # ordre de valors de carta
        idx = ordre.index(carta.numero)
        for i in ordre[:idx]:
            self.prob_mans_companys[jugador, carta.pal*12 + i - 1] = 0
    
    def __res_mes_gran_del_pal_que__(self, jugador: int, carta: Carta):
        ordre = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 1, 9] # ordre de valors de carta
        idx = ordre.index(carta.numero)
        for i in ordre[idx:]:
            self.prob_mans_companys[jugador, carta.pal*12 + i - 1] = 0
    
    def __ultima_carta_supera__(self) -> bool:
        if len(self.taula) == 2:
            if self.taula[1].get_valor(self.taula[0].pal, self.trumfo) > self.taula[0].get_valor(self.taula[0].pal, self.trumfo):
                return True
        if len(self.taula) == 3:
            if (self.taula[2].get_valor(self.taula[0].pal, self.trumfo) > self.taula[0].get_valor(self.taula[0].pal, self.trumfo)) and (self.taula[2].get_valor(self.taula[0].pal, self.trumfo) > self.taula[1].get_valor(self.taula[0].pal, self.trumfo)):
                return True
        return False

    def __trumfos_taula__(self) -> Tuple[bool, Carta]:
        trumfos_taula = []
        for c in self.taula:
            if c.pal == self.trumfo:
                trumfos_taula.append(c)
        return len(trumfos_taula) > 0, max(trumfos_taula)
    
    def __carta_mes_gran_del_pal__(self) -> Carta:
        del_pal = []
        for c in self.taula:
            if c.pal == self.taula[0].pal:
                del_pal.append(c)
        return max(del_pal)
    
    def update_state(self, ultim_jugador: int, action: int, proxim_jugador: int):
        carta_jugada = decode_action_card(action)
        # Actualitzem la probabilitat de tenir la carta jugada a 0 per a tothom
        for i in range(0, 4):
            self.prob_mans_companys[i, action] = 0                                              # cap jugador pot tenir ja la carta jugada
        
        if self.__basa_del_company__():                                                         # basa del conmpany
            if carta_jugada.pal == self.taula[0].pal:                                           # .. juga pal
                if carta_jugada.punts == 0:                                                     # .... no te punts
                    self.__res_mes_petit_del_pal_que__(ultim_jugador, carta_jugada)                     # ...... ha jugat la carta més petita d'aquell pal
            else:                                                                               # .. no juga pal
                for i in range(0, 12):                                                          # .... no te cap carta d'aquell pal              
                    self.prob_mans_companys[ultim_jugador, 12*self.taula[0].pal + i] = 0
                if carta_jugada.punts == 0:                                                     # .... no te punts
                    self.__res_mes_petit_del_pal_que__(ultim_jugador, carta_jugada)             # ...... ha jugat la carta més petita del pal que ha jugat    
        else:                                                                                   # no es basa del company 
            if carta_jugada.pal == self.taula[0].pal:                                           # .. juga pal
                if not self.__ultima_carta_supera__():                                          # .... NO supera les cartes jugades
                    self.__res_mes_petit_del_pal_que__(ultim_jugador, carta_jugada)             # ...... ha jugat la carta més petita del pal
                    hi_ha_trumfos, trumfo_mes_gran = self.__trumfos_taula__()
                    if not hi_ha_trumfos:                                                       # ...... si no hi ha trumfo a la taula
                        mes_gran_pal = self.__carta_mes_gran_del_pal__()                        
                        self.__res_mes_gran_del_pal_que__(ultim_jugador, mes_gran_pal)          # ........ no te res més gran que la carta del pal més gran jugada
            else:                                                                               # .. no juga pal
                for i in range(0, 12):                                                          # .... no te cap carta d'aquell pal              
                    self.prob_mans_companys[ultim_jugador, 12*self.taula[0].pal + i] = 0
                if not self.__ultima_carta_supera__():                                          # .... ultima carta tirada no guanya basa
                    hi_ha_trumfos, trumfo_mes_gran = self.__trumfos_taula__()                                           
                    if not hi_ha_trumfos:                                                       # ...... no hi ha trumfos a la taula
                        for i in range(0, 12):                                                  # ........ no te cap trumfo              
                            self.prob_mans_companys[ultim_jugador, 12*self.trumfo + i] = 0      
                    else:                                                                       # ...... hi ha trumfos a la taula
                        self.__res_mes_petit_del_pal_que__(ultim_jugador, carta_jugada)         # ........ ha jugat la carta més petita del pal
                        self.__res_mes_gran_del_pal_que__(ultim_jugador, trumfo_mes_gran)       # ........ no te cap trumfo més gran que trumfo mes gran taula

    def get_state(self, id_jugador: int):
        # Retorna l'estat actual del joc com un diccionari o una altra estructura de dades
        trumfo = one_hot_encode_trumfo_v2(self.trumfo)
        ma = one_hot_encode_hand(self.jugadors[id_jugador].ma)
        ma_altres = []
        for c in self.taula:
            ma_altres.extend(one_hot_encode_card(c))
        for i in range(3 - len(self.taula)):
            ma_a = copy(self.prob_mans_companys[((id_jugador + 1) % 4 + i) % 4])
            # fer una mascara posar a 0 les cartes de ma que valen 1 a self.jugadors[id_jugador].ma
            for carta in self.jugadors[id_jugador].ma:
                ma_a[carta.idx()] = 0
            ma_altres.extend(ma_a.tolist())
        return np.array(trumfo + ma + ma_altres, dtype=np.int8)