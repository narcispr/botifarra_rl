from botifarra.pals import BOTIFARRA
from botifarra.rl_utils import one_hot_encode_trumfo, one_hot_encode_hand, one_hot_encode_card, decode_action_card
from botifarra.botifarra import Botifarra
from botifarra.carta import Carta

from gymnasium import Env, spaces
import numpy as np
from copy import copy 
from typing import Tuple

class BotifarraEnv(Botifarra, Env):
    def __init__(self):
        Env.__init__(self)
        Botifarra.__init__(self)
        # Definim l'espai d'observació per compactar més la informació dels companys
        self.observation_space = spaces.Box(low=0, high=1, shape=(240,), dtype=np.int8)
      
        # Action space is an integer representing the index of the card to play
        self.action_space = spaces.Discrete(48)
      
        # hot encoding (0 o 1) indicant si cada jugador pot o no tenir aquella carta
        # segons el que s'ha jugat fins aquell moment seguint la botifarra obligada
        self.prob_mans_companys = np.ones((4, 48), dtype=int)
        
    def reset(self):
        # reset partida
        self.prob_mans_companys = np.ones((4, 48), dtype=int)
        self.reset_partida()
        info = {}
        info['proxim_jugador'] = self.jugador_actual + 1

        # Cantar 
        info['canta'] = self.jugador_inicial + 1
        self.trumfo = self.cantar_trumfo(self.jugador_inicial)
        info['trumfo'] = self.trumfo

        # Omple la màscara d'accions vàlides i el primer estat
        ma_valida = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
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

        # Actualitzem probabilitat cartes jugadors segon ultima carta jugada
        self.update_state(self.jugador_actual, action)
       

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
        ma_valida = self.jugadors[self.jugador_actual].cartes_valides(self.trumfo, self.taula)
        info['mask'] = one_hot_encode_hand(ma_valida)  # Màscara d'accions vàlides per al següent estat
        
        # Agafa l'estat pel nou jugador
        state = self.get_state(self.jugador_actual)

        return state, reward, done, terminated, info


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

        return len(trumfos_taula) > 0, max(trumfos_taula) if len(trumfos_taula) > 0 else None
    
    def __carta_mes_gran_del_pal__(self) -> Carta:
        del_pal = []
        for c in self.taula:
            if c.pal == self.taula[0].pal:
                del_pal.append(c)
        return max(del_pal)
    
    def update_state(self, ultim_jugador: int, action: int):
        carta_jugada = decode_action_card(action)
        
        # Actualitzem la probabilitat de tenir la carta jugada a 0 per a tothom
        for i in range(0, 4):
            self.prob_mans_companys[i, action] = 0                                              # cap jugador pot tenir ja la carta jugada
        
        if len(self.taula) == 1: # primer jugador de la basa
            return
                                              
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
                    hi_ha_trumfos, _ = self.__trumfos_taula__()
                    if not hi_ha_trumfos:                                                       # ...... si no hi ha trumfo a la taula
                        mes_gran_pal = self.__carta_mes_gran_del_pal__()                        
                        self.__res_mes_gran_del_pal_que__(ultim_jugador, mes_gran_pal)          # ........ no te res més gran que la carta del pal més gran jugada
            else:                                                                               # .. no juga pal
                for i in range(0, 12):                                                          # .... no te cap carta d'aquell pal              
                    self.prob_mans_companys[ultim_jugador, 12*self.taula[0].pal + i] = 0
                if not self.__ultima_carta_supera__():                                          # .... ultima carta tirada no guanya basa
                    hi_ha_trumfos, trumfo_mes_gran = self.__trumfos_taula__()                                           
                    if not hi_ha_trumfos and self.trumfo != BOTIFARRA:                                                       # ...... no hi ha trumfos a la taula
                        for i in range(0, 12):                                                  # ........ no te cap trumfo              
                            self.prob_mans_companys[ultim_jugador, 12*self.trumfo + i] = 0      
                    else:                                                                       # ...... hi ha trumfos a la taula
                        self.__res_mes_petit_del_pal_que__(ultim_jugador, carta_jugada)         # ........ ha jugat la carta més petita del pal
                        if self.trumfo != BOTIFARRA:                                            # ........ si hi ha trumfo i no es botifarra
                            self.__res_mes_gran_del_pal_que__(ultim_jugador, trumfo_mes_gran)   # .......... no te cap trumfo més gran que trumfo mes gran taula

    def get_state(self, id_jugador: int):
        # Retorna l'estat actual del joc com un diccionari o una altra estructura de dades
        trumfo = one_hot_encode_trumfo(self.trumfo)
        ma = one_hot_encode_hand(self.jugadors[id_jugador].ma)
        ma_altres = []
        for c in self.taula:
            ma_altres.extend(one_hot_encode_card(c))
        for i in range(3 - len(self.taula)):
            ma_a = copy(self.prob_mans_companys[(id_jugador + 1 + i) % 4])
            # fer una mascara posar a 0 les cartes de ma que valen 1 a self.jugadors[id_jugador].ma
            for carta in self.jugadors[id_jugador].ma:
                ma_a[carta.idx()] = 0
            ma_altres.extend(ma_a.tolist())
        return np.array(trumfo + ma + ma_altres, dtype=np.int8)
    
    def print_hist(self):
        for j in range(1):
            print(f"Jugador {j+1}")
            print(self.prob_mans_companys[j].reshape((4, 12)))
                  
            