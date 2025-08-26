from learning_lib.dqn import DQN
from typing import List
from gymnasium import Env, spaces
import torch as T
import numpy as np
from botifarra.rl_utils import decode_action_card

class DQNBotifarra(DQN):
    def __init__(self, input_dims=390, n_actions=48, batch_size=32, gamma=0.99, epsilon=1.0, 
                 lr=0.001, max_mem_size=10000000, eps_end=0.05, eps_dec=5e-6, hidden_layers=[256, 128]):
        super().__init__(input_dims, n_actions, batch_size=batch_size, gamma=gamma, epsilon=epsilon, 
                         lr=lr, max_mem_size=max_mem_size, eps_end=eps_end, eps_dec=eps_dec, hidden_layers=hidden_layers)
    
    def choose_action(self, observation, legal_actions_mask: np.ndarray, deterministic: bool=False) -> int:
        
        state = T.tensor(observation, dtype=T.float32, device=self.Q_eval.device)
        if self.dueling_dqn:
            state = state.unsqueeze(0)

        # Inference with no_grad (and eval() if you use BN/Dropout)
        self.Q_eval.eval()
        with T.no_grad():
            q_values = self.Q_eval(state).cpu().detach().numpy()
        self.Q_eval.train()

        # Filtrar les accions legals considerant que legal_actions_mask conté 48 valors on 0 vol dir acció ilegal i 1 acció legal
        # Les accions ilegals es posen a -inf per assegurar que no seran seleccionades
        filtered_q_values = np.where(legal_actions_mask, q_values, -np.inf)
        
        if np.random.random() > self.epsilon or deterministic:
            # pick best action
            action = np.argmax(filtered_q_values, axis=-1).item()
        else:
            # pick random legal action
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        return action
    

    def training(self, env, n_episodes, max_score=200):
        scores = []
        eps_history = []
        
        for i in range(n_episodes):
            if (i+1) % 1000 == 0:
                print(f"Episode {i+1}/{n_episodes}")
            if (i+1) % 5000 == 0:
                self.save_weights(f"./tmp_dqn_{i+1}")
            done = False
            observation, info = env.reset()
            jugades_fetes = 0
            while not done:
                # print(f"Jugada {jugades_fetes+1}/12")
                obs = np.zeros((5, 390), dtype=np.float32)
                act = np.zeros((4,), dtype=np.int32)
                obs[0] = observation
                with T.no_grad():
                    primer_jugador = (info['proxim_jugador'] - 1) % 4
                    for i in range(4):
                        # print(f"\tJugador {(primer_jugador + i) % 4 + 1}/4")
                        action = self.choose_action(observation, np.array(info['mask']))
                        # print(f"\t\tCarta jugada: {decode_action_card(action)}")
                        observation_, reward, terminated, truncated, info = env.step(action)
                        obs[i+1, :] = observation_
                        act[i] = action
                        observation = observation_
                    
                    if terminated or truncated:
                        done = True
                    jugades_fetes += 1
                    # print(f"Guanya el jugador {info['guanyador']}. Punts {reward}\n")

                    # Quan acaba la jugada guardem les 4 transicions al reply buffer
                    d = False
                    winner = info.get('guanyador')-1
                    for i in range(4):
                        if done and i == 3:
                            d = True
                        if (primer_jugador + winner) % 4 == i or (primer_jugador + winner + 2) % 4 == i:
                            r = reward
                        else:
                            r = -1 * reward
                        self.replay_buffer.store_transition(obs[i], act[i], r, obs[i+1], d)
                        # print(f"-> Transició: s {obs[i][:8]} a {decode_action_card(act[i])} r {r} s' {obs[i+1][:8]} d {d}")

                self.learn()
                observation = observation_
            eps_history.append(self.epsilon)
            