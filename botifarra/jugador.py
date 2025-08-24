from botifarra.carta import Carta
import random

class Jugador:
    """
    Un jugador amb una mà de cartes i estratègies senzilles per jugar.
    """
    def __init__(self, id: int):
        self.ma: list[Carta] = []
        self.id : int = id  # Identificador del jugador (0..3)

    def cantar(self, delegar: bool = False) -> int:
        """
        Retorna el pal (de 0 a 3) que canta el jugador.
        Per cantar botifarra retorna 4.
        Si delega retorna -1.
        """
        # De moment, retorna un valor aleator i de 0 a 4
        return random.choice([0, 1, 2, 3, 4])

    def __cartes_pal__(self, pal: int) -> list:
        return [carta for carta in self.ma if carta.pal == pal]
    
    def __trumfos__(self, trumfo: int) -> list:
        return [carta for carta in self.ma if carta.pal == trumfo]
    
    def __guanya_a__(self, cartes: list, carta: Carta, trumfo: int, pal_jugada: int) -> list:
        """
        Retorna totes les cartes de la llista `cartes` que guanyen a `carta`.
        """
        guanyadores = []
        for c in cartes:
            if c.get_valor(pal_jugada, trumfo) > carta.get_valor(pal_jugada, trumfo):
                guanyadores.append(c)
        return guanyadores
    
    def __hem_de_matar__(self, taula: list, matar_idx: int, del_pal: list, trumfos: list, trumfo: int) -> list:
        if len(del_pal) > 0:   
            # ... i tenim Pal de la carta jugada:
            del_pal_mata = self.__guanya_a__(del_pal, taula[matar_idx], trumfo, taula[0].pal)
            if len(del_pal_mata) > 0:
                # tornem totes les que poden matar.
                return del_pal_mata
            else:
                # tornem la més petita
                return [min(del_pal)]
        # ... i no tenim Pal de la carta jugada ... 
        if len(trumfos) > 0:
            # ...pero tenim trumfos: Tornem tots els trumfos que maten
            trumfo_mata = self.__guanya_a__(trumfos, taula[matar_idx], trumfo, taula[0].pal)
            if len(trumfo_mata) > 0:
                # tornem totes les que poden matar.
                return trumfo_mata
            else:
            # ... ni trumfos que matin: Tornem totes les cartes
                return self.ma
        # ... si no tenim ni pal ni trumfos: Tornem totes les cartes
        return self.ma
        
    def cartes_valides(self, trumfo: int, taula: list) -> list:
        """
        Retorna la llista de cartes que el jugador pot jugar segons les regles.
        """
        # Primer jugador: Pot tirar el que vulgui
        if len(taula) == 0:
            return self.ma
        
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
                    return del_pal
                else:
                    # Si no tenim cartes del pal, qualsevol és vàlida
                    return self.ma
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
                    # Si tenim cartes del pal, qualsevol del pal és vàlida
                    return del_pal
                else:
                    # Si no tenim cartes del pal, qualsevol és vàlida
                    return self.ma
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
        ma_valida = self.cartes_valides(trumfo, taula)
        carta = random.choice(ma_valida)
        print(f"Jugador {self.id + 1} té {self.ma} de les quals {ma_valida} es poden jugar i juga {carta}.")
        self.ma.remove(carta)
        return carta
    
    def __repr__(self):
        # retorna l'id del jugador i les cartes que té 
        return f"J{self.id + 1}: {self.ma}"