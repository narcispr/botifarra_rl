import numpy as np

##############################################
#           Replay Buffer                    #
##############################################

    
class ReplayBuffer():
    def __init__(self, max_size, input_shape, action_shape=None):
        """
        input_shape: tuple of ints, e.g. (obs_dim,) or (channels, height, width)
        action_shape: None for discrete, or tuple/int for continuous actions
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        # Unpack the state dimensions
        # If the user passes a single int, wrap it as a 1‑tuple
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        # Now input_shape is always a tuple
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        # Actions: either discrete or continuous
        if action_shape is None:
            self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        else:
            # ensure it's a tuple
            if isinstance(action_shape, int):
                action_shape = (action_shape,)
            self.action_memory = np.zeros((self.mem_size, *action_shape),
                                          dtype=np.float32)

        self.reward_memory   = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.legal_mask_memory = np.zeros((self.mem_size, 48), dtype=bool)
        self.next_legal_mask_memory = np.zeros((self.mem_size, 48), dtype=bool)
        

    def store_transition(self, state, action, reward, new_state, done, legal_mask, next_legal_mask):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index]     = state
        self.new_state_memory[index] = new_state
        self.action_memory[index]    = action
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done
        self.legal_mask_memory[index] = legal_mask
        self.next_legal_mask_memory[index] = next_legal_mask
        
        self.mem_cntr += 1

    def get_batch(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_idxs = np.random.choice(max_mem, batch_size, replace=False)
        states     = self.state_memory[batch_idxs]
        actions    = self.action_memory[batch_idxs]
        rewards    = self.reward_memory[batch_idxs]
        new_states = self.new_state_memory[batch_idxs]
        dones      = self.terminal_memory[batch_idxs]
        legal_masks = self.legal_mask_memory[batch_idxs]
        next_legal_masks = self.next_legal_mask_memory[batch_idxs]
        return states, actions, rewards, new_states, dones, legal_masks, next_legal_masks
    
