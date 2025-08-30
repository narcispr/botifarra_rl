from botifarra.carta import Carta
from botifarra.pals import OROS, COPES, ESPASES, BASTOS, BOTIFARRA

import random
from typing import List, Tuple

class Jugador:
    """
    Un jugador amb una mà de cartes i estratègies senzilles per jugar.
    """
    def __init__(self, id: int):
        self.ma: list[Carta] = []
        self.id : int = id  # Identificador del jugador (0..3)

    def ordenar_ma(self):
        # sort list ma by Carta.ordre 
        self.ma.sort(key=lambda carta: carta.ordre)
        
    def cantar(self, delegat: bool = False) -> int:
        """
        Retorna el pal (de 0 a 3) que canta el jugador.
        Per cantar botifarra retorna 4.
        Si delega retorna -1.
        """
        # De moment, retorna un valor aleator i de 0 a 4
        oros = self.__cartes_pal__(OROS)
        copes = self.__cartes_pal__(COPES)
        espases = self.__cartes_pal__(ESPASES)
        bastos = self.__cartes_pal__(BASTOS)
        # Si tenim 5 o més cartes d'algun pal, el cantem (no mirem el semi-fallo....)
        if (len(oros) >= 5) and (min(len(copes), len(espases), len(bastos)) <= 1):
            return OROS
        elif (len(copes) >= 5) and (min(len(oros), len(espases), len(bastos)) <= 1):
            return COPES
        elif (len(espases) >= 5) and (min(len(copes), len(oros), len(bastos)) <= 1):
            return ESPASES
        elif (len(bastos) >= 5) and (min(len(copes), len(espases), len(oros)) <= 1):
            return BASTOS
        
        # Si tenim molts trumfos cantem botifarra
        if sum(carta.get_punts() for carta in self.ma) > 20:
            return BOTIFARRA
        
        # Si no podem cantar però podem delegar, deleguem
        if not delegat:
            return -1
        else:
            # Si no podem delegar, cantem el pal que més cartes tenim
            cartes_pals = [len(oros), len(copes), len(espases), len(bastos)]
            max_cartes = max(cartes_pals)
            return cartes_pals.index(max_cartes)
        
    def __cartes_pal__(self, pal: int) -> List[Carta]:
        return [carta for carta in self.ma if carta.pal == pal]
    
    def __trumfos__(self, trumfo: int) -> List[Carta]:
        return [carta for carta in self.ma if carta.pal == trumfo]
    
    def __guanya_a__(self, cartes: List[Carta], carta: Carta, trumfo: int, pal_jugada: int) -> List[Carta]:
        """
        Retorna totes les cartes de la llista `cartes` que guanyen a `carta`.
        """
        guanyadores = []
        for c in cartes:
            if c.get_valor(pal_jugada, trumfo) > carta.get_valor(pal_jugada, trumfo):
                guanyadores.append(c)
        return guanyadores
    
    def __cartes_amb_punts__(self, pal: int = -1) -> List[Carta]:
        return [carta for carta in self.ma if carta.get_punts() > 0 and ((carta.pal == pal) or (pal == -1))]
    
    def __mes_petita_del_pal__(self, pal: int = -1) -> List[Carta]:
        cartes = []
        for p in range(4):
            if pal == -1 or p == pal:
                cartes_pal = self.__cartes_pal__(p)
                if len(cartes_pal) > 0:
                    cartes.append(min(cartes_pal))
        return cartes
    
    def __amb_punts_o_mes_petita__(self, pal: int=-1) -> List[Carta]:
        # Retorna les cartes __cartes_amb_punts__ + la __mes_petita_del_pal__. Utilitza un conjunt per evitar duplicats!
        cartes = set(self.__cartes_amb_punts__(pal=pal) + self.__mes_petita_del_pal__(pal=pal))
        return list(cartes)
       
    def __hem_de_matar__(self, taula: List[Carta], matar_idx: int, del_pal: List[Carta], trumfos: List[Carta], trumfo: int) -> Tuple[List[Carta], bool, bool]:
        falla_pal = False       # significa que no te més cartes d'aquell pal
        falla_trumfo = False    # significa que no te trumfo que marti la primera base (i.e., si qui guanya la base es trumfo és que no te trumfo però si es trumfo vol dir que no te cap carta superior al trumfo jugat)

        if len(del_pal) > 0:   
            # ... i tenim Pal de la carta jugada:
            del_pal_mata = self.__guanya_a__(del_pal, taula[matar_idx], trumfo, taula[0].pal)
            if len(del_pal_mata) > 0:
                # tornem totes les que poden matar.
                return del_pal_mata, falla_pal, falla_trumfo
            else:
                # tornem la més petita
                return [min(del_pal)], falla_pal, falla_trumfo
        # ... i no tenim Pal de la carta jugada ... 
        falla_pal = True
        if len(trumfos) > 0:
            # ...pero tenim trumfos: Tornem tots els trumfos que maten
            trumfo_mata = self.__guanya_a__(trumfos, taula[matar_idx], trumfo, taula[0].pal)
            if len(trumfo_mata) > 0:
                # tornem totes les que poden matar.
                return trumfo_mata, falla_pal, falla_trumfo
            else:
            # ... ni trumfos que matin: Tornem la mes petita de cada pal
                falla_trumfo = True
                return self.__mes_petita_del_pal__(), falla_pal, falla_trumfo
            
        # ... si no tenim ni pal ni trumfos: Tornem la més petita de cada pal (no matem...)
        falla_trumfo = True
        return self.__mes_petita_del_pal__(), falla_pal, falla_trumfo
        
    def cartes_valides(self, trumfo: int, taula: list) -> Tuple[List[Carta], bool, bool]:
        """
        Retorna la llista de cartes que el jugador pot jugar segons les regles.
        També torna 2 bolleans indican si fall pal i si falla trumfo en cas que amb
        el que ha tirat és pugui saber, altrament False.
        """
        falla_pal = False
        falla_trumfo = False

        # Primer jugador: Pot tirar el que vulgui
        if len(taula) == 0:
            return self.ma, falla_pal, falla_trumfo
        
        trumfos = self.__trumfos__(trumfo)
        del_pal = self.__cartes_pal__(taula[0].pal)
        
        # Segon...
        if len(taula) == 1:
            # ... hem de matar si podem
            return self.__hem_de_matar__(taula, 0, del_pal, trumfos, trumfo)
            
        # Tercer...
        elif len(taula) == 2:
            # ... i la mà és del company:
            if (taula[0].get_valor(taula[0].pal, trumfo) > taula[1].get_valor(taula[0].pal, trumfo)):
                if len(del_pal) > 0:
                    # Si tenim cartes del pal, qualsevol del pal és vàlida
                    return self.__amb_punts_o_mes_petita__(taula[0].pal), falla_pal, falla_trumfo
                else:
                    # Si no tenim cartes del pal, qualsevol és vàlida
                    falla_pal = True
                    return self.__amb_punts_o_mes_petita__(), falla_pal, falla_trumfo
            # ... i la mà no és del company:
            else:
                # Hem de matar si podem
                return self.__hem_de_matar__(taula, 1, del_pal, trumfos, trumfo)
        # Quart...
        elif len(taula) == 3:
            # ... i la mà és del company:
            if (taula[1].get_valor(taula[0].pal, trumfo) > 
                taula[0].get_valor(taula[0].pal, trumfo) and
                taula[1].get_valor(taula[0].pal, trumfo) > 
                taula[2].get_valor(taula[0].pal, trumfo)):
                if len(del_pal) > 0:
                    # Si tenim cartes del pal, qualsevol del pal amb punts o la més petita és vàlida
                    return self.__amb_punts_o_mes_petita__(taula[0].pal), falla_pal, falla_trumfo
                else:
                    # Si no tenim cartes del pal, qualsevol del pal amb punts o la més petita és vàlida
                    falla_pal = True
                    return self.__amb_punts_o_mes_petita__(), falla_pal, falla_trumfo
            # ... i la mà no és del company:
            else:
                # Si podem, hem de matar la carta més alta dels jugadors rivals
                idx_a_matar = 0
                if (taula[2].get_valor(taula[0].pal, trumfo) > 
                    taula[0].get_valor(taula[0].pal, trumfo)):
                    idx_a_matar = 2
                return self.__hem_de_matar__(taula, idx_a_matar, del_pal, trumfos, trumfo)
        else:
            raise ValueError("La taula ja té més de 3 cartes!")

    def jugar(self, trumfo: int, taula: list) -> Carta:
        ma_valida, falla_pal, falla_trumfo = self.cartes_valides(trumfo, taula)
        carta = random.choice(ma_valida)
        print(f"Jugador {self.id + 1} té {self.ma} de les quals {ma_valida} es poden jugar i juga {carta}. (falla_pal={falla_pal}, falla_trumfo={falla_trumfo})")
        self.ma.remove(carta)
        return carta
    
    def __repr__(self):
        # retorna l'id del jugador i les cartes que té 
        return f"J{self.id + 1}: {self.ma}"