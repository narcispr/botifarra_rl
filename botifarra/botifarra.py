from botifarra.jugador import Jugador
from botifarra.baralla import Baralla

class Botifarra:
    """
    Motor simplificat d'una partida de botifarra.
    """

    def __init__(self):
        self.jugadors: list[Jugador] = [Jugador(i) for i in range(4)]
        self.punts_equip_a = 0
        self.punts_equip_b = 0
        self.pals = {0: 'Oros', 1: 'Copes', 2: 'Espases', 3: 'Bastos', 4: 'Botifarra'}

    def __carta_guanyadora__(self, trumfo: int, taula: list) -> int:
        """
        Retorna l'índex (0..3) de la carta guanyadora de la taula.
        """
        valors = []
        for i in range(4):
            valors.append(taula[i].get_valor(taula[0].pal, trumfo))
        return valors.index(max(valors))
    
    def jugada(self, jugador_inicial: int, trumfo: int):
        """
        Simula una jugada completa (4 cartes) començant pel jugador `jugador_inicial`
        i amb el pal de trumfo `trumfo`.
        Retorna l'índex del jugador que guanya la jugada (0..3) i els punts de la mà.
        """
        taula = []
        for i in range(4):
            idx_jugador = (jugador_inicial + i) % 4
            carta = self.jugadors[idx_jugador].jugar(trumfo, taula)
            taula.append(carta)
          
        # Determinar qui guanya la jugada
        idx_guanyador = self.__carta_guanyadora__(trumfo, taula)
        guanyador = (jugador_inicial + idx_guanyador) % 4
        punts_jugada = sum(carta.get_punts() for carta in taula) + 1 # + 1 punt per cada jugada
       
        print(f"Jugador {guanyador + 1} guanya la jugada i fa {punts_jugada} punts.")
        return guanyador, punts_jugada
    
    def repartir_cartes(self):
        # Repartir cartes
        baralla = Baralla()
        baralla.barreja()
        for i in range(4):
            self.jugadors[i].ma = baralla.reparteix(12)
            # print(f"{self.jugadors[i]}")
    
    def cantar_trumfo(self, jugador_inicial: int) -> int:
        # Cantar 
        trumfo = self.jugadors[jugador_inicial].cantar()
        if trumfo == -1:
            # print(f"El jugador {jugador_inicial + 1} ha delegat el cant.")
            trumfo = self.jugadors[(jugador_inicial + 2) % 4].cantar(delegat=True)
            # print(f"El jugador {(jugador_inicial + 2) % 4 + 1} canta el pal {self.pals[trumfo]}.")
        else:
            # print(f"El jugador {jugador_inicial + 1} canta el pal {self.pals[trumfo]}.")
            pass
        
        return trumfo
    
    def jugar_ma(self, jugador_inicial: int):
        """
        Simula una mà completa de botifarra (12 jugades).
        """
        punts_a = 0
        punts_b = 0
        # Repartir cartes
        self.repartir_cartes()
        
        # Cantar 
        trumfo = self.cantar_trumfo(jugador_inicial)
        
        # Simular les 12 jugades
        j = (jugador_inicial + 1) % 4
        for ronda in range(12):
            print(f"\nRonda {ronda + 1}:")
            guanyador, punts_jugada = self.jugada(j, trumfo)
            if guanyador % 2 == 0:
                punts_a += punts_jugada
            else:
                punts_b += punts_jugada
            j = guanyador
        
        if punts_a > punts_b:
            self.punts_equip_a += (punts_a - 36)
        elif punts_b > punts_a:
            self.punts_equip_b += (punts_b - 36)
        else:
            print("Empat a punts en la mà!")

        print(f"\Equip A {punts_a} punts contra {punts_b} de l'Equip B.")
        print(f"\nPunts final Equip A: {self.punts_equip_a}, Equip B: {self.punts_equip_b}")

    def jugar_partida(self):
        """
        Simula una partida completa de botifarra fins que algun equip superi els 100 punts.
        """
        self.punts_equip_a = 0
        self.punts_equip_b = 0
        j = 0  # jugador inicial
        numero_ma = 1
        while self.punts_equip_a < 100 and self.punts_equip_b < 100:
            print(f"\nIniciant mà número {numero_ma}. Punts actuals - Equip A: {self.punts_equip_a}, Equip B: {self.punts_equip_b}")
            self.jugar_ma(jugador_inicial=j)
            j = (j + 1) % 4
            numero_ma += 1
        if self.punts_equip_a >= 100:
            print("\nL'Equip A guanya la partida!")
        else:
            print("\nL'Equip B guanya la partida!")
