from botifarra.botifarra_env import BotifarraEnv
from botifarra.dqn_botifarra import DQNBotifarra

agent = DQNBotifarra()

env = BotifarraEnv()
agent.training(env, 1000000, save_every=10000, log_every=2000)
agent.save_weights("./botifarra_1M_dqn")
