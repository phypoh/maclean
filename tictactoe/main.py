import numpy as np

from board_rules import *
from bot_class import Bot
import pickle

# pvp()

def save_bot(bot, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(bot, file)

def load_bot(filepath):
    with open(filepath, 'rb') as file:
        bot = pickle.load(file)

    return bot

# bot = Bot(0.2, 3)
# bot.train(100000)
# save_bot(bot, 'bot.txt')

bot = load_bot('bot.txt')

while True:
    pve(bot)

