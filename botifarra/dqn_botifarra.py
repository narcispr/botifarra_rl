from botifarra.rl_utils import decode_action_card
from botifarra.nn_architecture import CardDQN
from botifarra.replaybuffer import ReplayBuffer

from typing import List
from gymnasium import Env, spaces
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQNBotifarra():
    def __init__(
        self,
        input_dims: int = 240,          
        n_actions: int = 48,
        batch_size: int = 64,           
        gamma: float = 0.99,
        epsilon: float = 1.0,
        lr: float = 1e-3,
        max_mem_size: int = 1_000_000,
        eps_end: float = 0.05,
        eps_dec: float = 5e-6,
        # Paràmetres de la CardDQN
        d: int = 128,
        head_hidden: int = 128,
        use_transformer: bool = True,
        n_layers: int = 1,
        n_heads: int = 4,
        # Optimització
        grad_clip_norm: float = 1.0,    # CHANGE: clipping evita explosions de gradient
        huber_delta: float = 1.0        # CHANGE: Huber (SmoothL1) és més estable que MSE
    ):
        # Set agent parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.grad_clip_norm = grad_clip_norm
        self.huber_delta = huber_delta

        # Espai d'accions (llista 0..47)
        self.action_space = [i for i in range(self.n_actions)]
        
        # Set replay buffer
        self.replay_buffer = ReplayBuffer(max_mem_size, input_dims)
        
        # Set Q network
        self.Q_eval = CardDQN(
            d=d, head_hidden=head_hidden,
            use_transformer=use_transformer, n_layers=n_layers, n_heads=n_heads
        )
        self.Q_target = CardDQN(
            d=d, head_hidden=head_hidden,
            use_transformer=use_transformer, n_layers=n_layers, n_heads=n_heads
        )
    
        # Set target Q network and hard copy weights from Q network
        self.Q_target.update_weights(self.Q_eval, soft=False)

        # Set device
        self.device = next(self.Q_eval.parameters()).device  # no dependre d’un atribut .device

        # print Q target architecture
        print(self.Q_target)
        # print number of parameters in Q target
        print(f"Number of parameters in Q target: {sum(p.numel() for p in self.Q_target.parameters() if p.requires_grad)}")
        
        # Set optimizer and loss function
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        # Huber (SmoothL1) -> més robust a outliers del TD error
        self.loss_fn = nn.SmoothL1Loss(beta=self.huber_delta)

    @T.no_grad()
    def choose_action(self, observation: np.ndarray, legal_actions_mask: np.ndarray, deterministic: bool=False) -> int:
        
        state = T.tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)  # (1, 240) o (1,48,5)
        legal_mask = T.tensor(legal_actions_mask.astype(bool), dtype=T.bool, device=self.device).unsqueeze(0)  # (1,48)

        if deterministic or (np.random.random() > self.epsilon):
            # pick best action
            self.Q_eval.eval()
            q_values = self.Q_eval(state, legal_mask=legal_mask)
            action = int(q_values.argmax(dim=-1).item())
            self.Q_eval.train()
        else:
            # pick random legal action
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        return action
    
    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return
        
        # Reset the gradients
        self.optimizer.zero_grad()

        # Get a batch of transitions
        states, action_batch, rewards, new_states, dones, legal_masks, next_legal_masks = self.replay_buffer.get_batch(self.batch_size)
        
        # Transform arrays to torch tensors
        state_batch             = T.tensor(states, dtype=T.float32, device=self.device)          # (B,240) o (B,48,5)
        new_state_batch         = T.tensor(new_states, dtype=T.float32, device=self.device)      # idem
        reward_batch            = T.tensor(rewards, dtype=T.float32, device=self.device)         # (B,)
        done_batch              = T.tensor(dones, dtype=T.bool, device=self.device)              # (B,)
        action_batch_t          = T.tensor(action_batch, dtype=T.long, device=self.device)       # (B,)
        legal_mask_batch        = T.tensor(legal_masks, dtype=T.bool, device=self.device)      # (B,48)
        next_legal_mask_batch   = T.tensor(next_legal_masks, dtype=T.bool, device=self.device) # (B,48)
    
        # Zero grads
        self.optimizer.zero_grad(set_to_none=True)  # set_to_none millora rendiment/memòria

        # Get the Q values for the current state using Q and the next state using Q target, 
        # set next to 0 if terminal
        q_eval_all = self.Q_eval(state_batch, legal_mask=legal_mask_batch)  # (B,48)
        # Seleccionem Q(s, a) amb gather 
        q_eval = q_eval_all.gather(1, action_batch_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN: a' = argmax_a Q_eval(s', a) amb màscara d'accions legals si la tenim
        with T.no_grad():  # Evitem backprop pel camí target
            q_next_eval_all = self.Q_eval(new_state_batch, legal_mask=next_legal_mask_batch)  # (B,48)
            if next_legal_mask_batch is not None:
                q_next_eval_all = CardDQN.apply_action_mask(q_next_eval_all, next_legal_mask_batch)
            next_best_action = q_next_eval_all.argmax(dim=1)  # (B,)

            # Q_target(s', a')
            q_next_target_all = self.Q_target(new_state_batch, legal_mask=next_legal_mask_batch)  # (B,48)
            if next_legal_mask_batch is not None:
                q_next_target_all = CardDQN.apply_action_mask(q_next_target_all, next_legal_mask_batch)
            q_next = q_next_target_all.gather(1, next_best_action.unsqueeze(1)).squeeze(1)  # (B,)
            q_next = q_next.masked_fill(done_batch, 0.0)  # (B,)

            # Objectiu TD
            q_target = reward_batch + self.gamma * q_next  # (B,)        

        # Calculate the loss and backpropagate
        loss = self.loss_fn(q_eval, q_target)
        loss.backward()
        # gradient clipping per estabilitat
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.Q_eval.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Decrease epsilon if it is above the minimum
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_dec) 
        
        # Update the target network
        self.Q_target.update_weights(self.Q_eval)

    def training(self, env, n_episodes, save_every: int = 5000, log_every: int = 1000):
        scores = []
        eps_history = []
        
        for ep in range(n_episodes):
            if (ep + 1) % log_every == 0:
                print(f"Episode {ep + 1}/{n_episodes}")
            if (ep + 1) % save_every == 0:
                self.save_weights(f"./tmp_dqn_{ep + 1}")
            
            done = False
            observation, info = env.reset()
            
            jugades_fetes = 0
            while not done:
                # print(f"Jugada {jugades_fetes+1}/12")
                obs_hist = np.zeros((5, self.input_dims), dtype=np.float32)
                mask_hist = np.zeros((5, self.n_actions), dtype=np.uint8)
                act_hist = np.zeros((4,), dtype=np.int32)
                obs_hist[0] = observation
                mask_hist[0] = np.array(info['mask'], dtype=np.uint8)

                primer_jugador = (info['proxim_jugador'] - 1) % 4
                for t in range(4):
                    action = self.choose_action(observation, np.array(info['mask']))
                    observation_, reward, terminated, truncated, info = env.step(action)

                    # Desar història
                    obs_hist[t + 1] = observation_
                    mask_hist[t + 1] = np.array(info['mask'], dtype=np.uint8)
                    act_hist[t] = action
                    observation = observation_
                
                done = terminated or truncated
                jugades_fetes += 1
                
                # Quan acaba la jugada guardem les 4 transicions al reply buffer
                winner = int(info.get('guanyador')) - 1
                for i in range(4):
                    d = bool(done and i == 3) # només la darrera de la mà tanca episodi
                    if (primer_jugador + winner) % 4 == i or (primer_jugador + winner + 2) % 4 == i:
                        r = float(reward)
                    else:
                        r = float(-1 * reward)

                    self.replay_buffer.store_transition(obs_hist[i], act_hist[i], r, obs_hist[i+1], d, mask_hist[i], mask_hist[i+1])
  
                self.learn()
                eps_history.append(self.epsilon)
            
    def save_weights(self, filename):
        print('... saving weights ...')
        T.save(self.Q_eval.state_dict(), filename + '.weights')
        
    def load_weights(self, filename):
        print('... loading weights ...')
        self.Q_eval.load_state_dict(T.load(filename + '.weights'))
        self.Q_target.load_state_dict(T.load(filename + '.weights'))