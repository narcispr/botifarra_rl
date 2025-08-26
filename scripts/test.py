# test the agent
from botifarra.botifarra_env import BotifarraEnv
from botifarra.rl_utils import decode_action_card
from botifarra.dqn_botifarra import DQNBotifarra

import numpy as np

NUMERO_PARTIDES = 100
PESOS_EQUIP_A = "/home/narcis/catkin_ws/src/botifarra/agents/botifarra_10k_dqn"
PESOS_EQUIP_B = "/home/narcis/catkin_ws/src/botifarra/agents/botifarra_50k_dqn"

agent_A = DQNBotifarra(hidden_layers=[128, 128])
agent_A.load_weights(PESOS_EQUIP_A)
agent_B = DQNBotifarra(hidden_layers=[512, 256])
agent_B.load_weights(PESOS_EQUIP_B)
env = BotifarraEnv()

victories_a = 0
victories_b = 0
total_punts_a = 0
total_punts_b = 0

for partides in range(NUMERO_PARTIDES):
    print(f"Partida {partides+1}/{NUMERO_PARTIDES}. Victories equip A: {victories_a} - Victories equip B: {victories_b}")
    punts_equip_a = 0
    punts_equip_b = 0
    ma_numero = 0
    while punts_equip_a < 100 and punts_equip_b < 100:
        # print(f"\nMa número {ma_numero + 1}. Punts actuals - Equip A: {punts_equip_a}, Equip B: {punts_equip_b}\n")
        obs, info = env.reset()
        # print(f"Jugador inicial per cantar: {info['canta']}")
        # for i in range(4):
        #     print(f"Jugador {i+1} té les cartes: {env.jugadors[i]}")

        # print(f"\n-> Trumfo: {env.pals[env.trumfo]}\n\n ")
        done = False
        while not done:
            # L'agent selecciona la carta a jugar (d'entre les vàlides!, d'aquí el mask)
            if info['proxim_jugador'] == 0 or info['proxim_jugador'] == 2:
                agent = agent_A
            else:
                agent = agent_B
            action = agent.choose_action(obs, np.array(info['mask']), deterministic=True)
            # print(f"Jugador {(info['proxim_jugador']-1) % 4 + 1} Carta jugada: {decode_action_card(action)}")
            state, reward, done, _, info = env.step(action)
            # if "guanyador" in info:
            #     print(f" --> Guanya el jugador {info['guanyador']}. Punts Jugada {reward}")
                # print(f" --> Punts actuals - Equip A: {env.punts_equip_a}, Equip B: {env.punts_equip_b}\n")

        # print(f"\nFi de la mà {ma_numero + 1}. Punts obtinguts: Equip A {info['punts_equip_a']}, Equip B {info['punts_equip_b']}\n")
        punts_equip_a += info['punts_equip_a']
        punts_equip_b += info['punts_equip_b']
        ma_numero += 1
        
    if punts_equip_a >= 100:
        # print(f"\nGuanya l'Equip A amb {punts_equip_a} punts contra {punts_equip_b} de l'Equip B!")
        victories_a += 1
    else:
        # print(f"\nGuanya l'Equip B amb {punts_equip_b} punts contra {punts_equip_a} de l'Equip A!") 
        victories_b += 1
    total_punts_a += punts_equip_a
    total_punts_b += punts_equip_b

if victories_a > victories_b:
    print(f"\nGuanya l'Equip A la sèrie amb {victories_a} victòries contra {victories_b} de l'Equip B!")
elif victories_b > victories_a:
    print(f"\nGuanya l'Equip B la sèrie amb {victories_b} victòries contra {victories_a} de l'Equip A!")
else:
    print(f"\nEmpat a victòries {victories_a} - {victories_b}!")
print(f"\nPunts totals Equip A: {total_punts_a}, Equip B: {total_punts_b}")