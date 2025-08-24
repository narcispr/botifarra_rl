import random
from botifarra.carta import Carta

class Baralla:
    """
    Construeix i gestiona la baralla espanyola.
    """
    def __init__(self):
        self.cartes = []
        self.reset()

    def reset(self):
        pals = list(range(4))
        nums = list(range(1, 13))  # 1..12
        self.cartes = [Carta(p, n) for p in pals for n in nums]

    def barreja(self):
        random.shuffle(self.cartes)

    def reparteix(self, cartes_per_jugador: int = 12):
        """
        Posa les N cartes (on N = `cartes_per_jugador`) de la part
        superior de la baralla a una llista anomenada `ma` i les treu de la baralla.
        """
        ma = self.cartes[:cartes_per_jugador]
        self.cartes = self.cartes[cartes_per_jugador:]
        return ma
