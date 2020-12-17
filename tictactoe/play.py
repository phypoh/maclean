import numpy as np

from board_rules import *
from q_bot_class import Bot
import pickle


def load_bot(filepath):
    with open(filepath, 'rb') as file:
        bot = pickle.load(file)

    return bot

bot = load_bot('q_bot.txt')

while True:
    pve(bot)

