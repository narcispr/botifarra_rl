import numpy as np

class AgentBotifarra():
    def __init__(self):
        pass

    def choose_action(self, state, legal_actions):
        # return the index of a random legal action
        legal_action_indices = np.where(legal_actions == 1)[0]
        return np.random.choice(legal_action_indices)