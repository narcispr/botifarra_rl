from botifarra.carta import Carta

# Donat un trumfo (0 a 3) codificar-lo amb un one-hot de 4 elements
def one_hot_encode_trumfo(trumfo: int) -> list:
    encoding = [0] * 4  # Suposant 4 pals
    if 0 <= trumfo < 4:
        encoding[trumfo] = 1
    return encoding

# Donada una llista de cartes codificarles amb ub hotbit (suposant 48 cartes, 1 si la carta esta a la llista 0 si no)
def one_hot_encode_hand(cartes: list) -> list:
    encoding = [0] * 48  # Suposant una baralla de 48 cartes
    for carta in cartes:
        index = carta.pal * 12 + (carta.numero - 1)  # Càlcul de l'índex basat en pal i número
        encoding[index] = 1
    return encoding

# Donada una carta codificada amb un onehot de 48 elements, retornar la carta corresponent
def decode_one_hot_card(encoding: list) -> Carta:
    index = encoding.index(1)  # Trobar l'índex on hi ha el 1
    pal = index // 12
    numero = (index % 12) + 1
    return Carta(pal, numero)

# Donada una carta codificada amb una acció (int de 0 a 47), retornar la carta corresponent
def decode_action_card(action: int) -> Carta:
    pal = action // 12
    numero = (action % 12) + 1
    return Carta(pal, numero)

def code_card(carta: Carta) -> int:
    return carta.pal * 12 + (carta.numero - 1)

def one_hot_encode_taula(taula: list) -> list:
    # Per cada carta a la taula, codificar-la amb la funció one-hot_encode_hand.
    # Concatenar totes les codificacions en una sola llista. Si la taula té menys de
    # 3 cartes, omplir amb 48 zeros més per cada carta que falta.
    encoding = []
    for carta in taula:
        encoding.extend(one_hot_encode_hand([carta]))
    for i in range(3 - len(taula)):
        encoding.extend([0] * 48)
    return encoding

def encode_estat(trumfo: int, ma: list, taula: list, ) -> list:
    # Codificar l'estat del joc amb la codificació del trumfo, la mà del jugador,
    # les cartes a la taula.
    # L'historic es codifica a un altre lloc.
    encoding = []
    encoding.extend(one_hot_encode_trumfo(trumfo))
    encoding.extend(one_hot_encode_hand(ma))
    encoding.extend(one_hot_encode_taula(taula))
    return encoding
    