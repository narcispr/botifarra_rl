from botifarra.agent import AgentBotifarra
from botifarra.dqn_botifarra import DQNBotifarra

class DQNAgent(AgentBotifarra):
    def __init__(self, weights_path):
        super().__init__()
        self.dqn = DQNBotifarra()
        self.dqn.load_weights(weights_path)

    def choose_action(self, state, legal_actions, deterministic=False):
        return self.dqn.choose_action(observation=state, legal_actions_mask=legal_actions, deterministic=True)