from botifarra.botifarra_env import BotifarraEnv
from botifarra.rl_utils import decode_action_card
import numpy as np

import random

if __name__ == "__main__":
    np.set_printoptions(
        linewidth=200,   # max number of characters per line before wrapping
        threshold=10000 # show all elements, don't abbreviate with '...'
    )

    env = BotifarraEnv()
    state, info = env.reset()
    print(f"state init:\n {state.reshape(5,48)}")
    print(f"\ninfo init:\n {info}")
    done = False
    
    while not done:
        # Selecciona aleatoriament lindex de un '1' dins del vector info['mask'] que és de l'estil [0,0, 0, 1, 0, 0, 1, 0, ...]
        idx_valids = [i for i, v in enumerate(info['mask']) if v == 1]
        action = random.choice(idx_valids)
        print(f"\n\nCarta jugada: {decode_action_card(action)}\n\n")
        state, reward, done, terminated, info = env.step(action)
        print(f"state:\n {state.reshape(5,48)}")
        print(f"\nreward:\n {reward}")
        print(f"\ninfo:\n {info}")
        