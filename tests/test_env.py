from botifarra.botifarra_env import BotifarraEnv
from botifarra.rl_utils import decode_action_card

import random

if __name__ == "__main__":
    env = BotifarraEnv()
    state, info = env.reset()
    print(f"Estat inicial: {state}\n\nInfo: {info}")
    done = False

    while not done:
        # Selecciona aleatoriament lindex de un '1' dins del vector info['mask'] que és de l'estil [0,0, 0, 1, 0, 0, 1, 0, ...]
        idx_valids = [i for i, v in enumerate(info['mask']) if v == 1]
        action = random.choice(idx_valids)
        print(f"\n\nCarta jugada: {decode_action_card(action)}\n\n")
        state, reward, done, terminated, info = env.step(action)
        print(f"Estat: {state}\n\nRecompensa\n\n{reward}\n\nFet: {done}\n\nInfo: {info}")