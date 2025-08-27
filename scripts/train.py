from botifarra.botifarra_env import BotifarraEnv
from botifarra.botifarra_env_v2 import BotifarraEnvV2

from botifarra.dqn_botifarra import DQNBotifarra

agent = DQNBotifarra()

env = BotifarraEnvV2()
agent.training(env, 10000)
agent.save_weights("./botifarra_v2_10k_dqn")