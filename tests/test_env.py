from botifarra.botifarra_env_v2 import BotifarraEnvV2
from botifarra.rl_utils import decode_action_card

import random

if __name__ == "__main__":
    env = BotifarraEnvV2()
    state, info = env.reset()
    print(f"Estat inicial: {state}\n\nInfo: {info}")
    done = False

    while not done:
        # Selecciona aleatoriament lindex de un '1' dins del vector info['mask'] que és de l'estil [0,0, 0, 1, 0, 0, 1, 0, ...]
        idx_valids = [i for i, v in enumerate(info['mask']) if v == 1]
        action = random.choice(idx_valids)
        print(f"\n\nCarta jugada: {decode_action_card(action)}\n\n")
        state, reward, done, terminated, info = env.step(action)
        print(f"Estat: {state[0:4]}\n{state[4:52]}\n{state[52:100]}\n{state[100:148]}\n{state[148:196]}\n\nRecompensa\n\n{reward}\n\nFet: {done}\n\nInfo: {info}")