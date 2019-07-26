import numpy as np

from board_rules import *
from bot_class import Bot

# pvp()

bot = Bot(0.2, 3)

bot.train(10000)

while True:
    pve(bot)

