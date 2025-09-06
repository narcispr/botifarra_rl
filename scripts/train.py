from botifarra.botifarra_env import BotifarraEnv
from botifarra.dqn_botifarra import DQNBotifarra

agent = DQNBotifarra()

env = BotifarraEnv()
agent.training(env, 1000, log_every=100)
agent.save_weights("./botifarra_v2_10k_dqn")