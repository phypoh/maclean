import numpy as np

from board_rules import *
from q_bot_class import Bot
import pickle

# pvp()

def save_bot(bot, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(bot, file)

bot = Bot(0.2, 3)
bot.train(100000)
save_bot(bot, 'q_bot.txt')
