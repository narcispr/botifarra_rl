from botifarra.botifarra_env import BotifarraEnv
from botifarra.dqn_botifarra import DQNBotifarra

agent = DQNBotifarra(batch_size=128)

env = BotifarraEnv()
agent.training(env, 50000)
agent.save_weights("botifarra_50k_dqn")