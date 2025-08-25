class Carta:
    """
    Representa una carta de la baralla espanyola (4 pals x 12 valors).
    """
    def __init__(self, pal: int, numero: int):
        # pal ∈ {'oros','copes','espases','bastos'}  # (només referencial)
        # numero ∈ {1,2,3,4,5,6,7,8,9,10,11,12}
        self.pal = pal
        self.numero = numero
        self.nom_pal = {0: 'O', 1: 'C', 2: 'E', 3: 'B'}.get(pal, 'desconegut')

        # Puntuació estàndard de botifarra (no-trumfo i trumfo)
        # 1 (as)=11, 3=10, 12 (rei)=4, 11 (cavall)=3, 10 (sota)=2, resta=0
        self.punts = {1: 4, 9: 5, 10: 1, 11: 2, 12: 3}.get(numero, 0)
        self.valor = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 1: 11, 9: 12}.get(numero, 0)

    def __repr__(self):
        return f"{self.numero}{self.nom_pal}"
    
    def __str__(self):
        return f"{self.numero}{self.nom_pal}"
    
    def __eq__(self, other):
        return self.pal == other.pal and self.numero == other.numero
    
    def __hash__(self):
        return hash((self.pal, self.numero))
    
    def __lt__(self, other):
        return (self.valor) < (other.valor)
    
    def get_punts(self):
        return self.punts
    
    def get_valor(self, pal_jugada: int, trumfo: int):
        bonus = 0
        if self.pal == pal_jugada:
            bonus += 12
        if self.pal == trumfo:
            bonus += 24
        return self.valor + bonus