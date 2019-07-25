import numpy as np


def init_board(length):
    """Initialise an empty board"""
    board = np.zeros((length,length), dtype=int)
    return board


def place_piece(board, position, side):
    """Place a piece on the board"""
    position = tuple(x - 1 for x in position)
    if board[position] != 0:
        print('Error! There is already a X or O in the location')
        return board, False
    else:
        board[position] = side
        return board, True

def disp_board(board):
    """Displays the board"""
    board = np.matrix(board, dtype=str)

    board[board == '1'] = 'X'
    board[board == '-1'] = 'O'
    board[board == '0'] = '_'

    print(board)


def check_win(board):
    """Check if any player has won"""
    length = len(board)
    check_row = board.sum(axis=1)
    check_col = board.sum(axis=0)

    if length in check_row or length in check_col or np.trace(board) == length or np.trace(np.fliplr(board)) == length:
        print('X wins')
        return 1

    if -length in check_row or -length in check_col or np.trace(board) == -length or np.trace(np.fliplr(board)) == -length:
        print('O wins')
        return -1

    if 0 not in board:
        print('Draw!')
        return 2

    return 0

def human_play(board, side):
    if side == 1:
        side_letter = 'X'
    elif side == -1:
        side_letter = 'O'
    print('Your turn: ', side_letter)
    position_input = input("Enter your piece position _, _: ")

    try:
        position = (int(position_input[0]), int(position_input[-1]))
        board, err_check = place_piece(board, position, side)
        if err_check == False:
            human_play(board, side)

    except:
        print("Wrong format! Try again")
        human_play(board, side)

    return

def pvp():
    board = init_board(3)
    win = 0
    side = 1

    while win == 0:
        human_play(board, side)
        disp_board(board)
        side = -side
        win = check_win(board)
    return