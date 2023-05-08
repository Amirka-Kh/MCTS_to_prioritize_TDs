"""
An example implementation of the abstract Node class for use in MCTS
If you run this file then you can play against the computer.

A prioritization problem domain is represented as a tuple of 4 values (TDs),
each either None or Int, respectively meaning 'empty', 'Reward'.

the problem domain as tuples:
(TD1, TD2, TD3, TD4)
"""

from collections import namedtuple
from random import choice
from monte_carlo_tree_search import MCTS, Node

_TTTB = namedtuple("Prioritizer", "tup state terminal")


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Prioritizer(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If the prioritization is finished then no TD refactorings can be made
            return set()
        # Otherwise, you can refactor not addressed TD
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value['addressed'] is None
        }

    def find_random_child(board):
        if board.terminal:
            return None  # If the prioritization is finished then no refactorings can be made
        empty_spots = [i for i, value in enumerate(board.tup) if value['addressed'] is False]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on nonterminal board {board}")
        st = board.state

        reward = st['lines'] / (st['statements'] + st['functions'] + st['classes'] + st['files'] + st['comments'])
        reward += (4 - st['issues']) / 4 + (st['debt_maintain'] + st['ref_eff_rel']) / 13
        return reward

    def is_terminal(board):
        return board.terminal  # returns boolean

    def make_move(board, index):
        tup = board.tup
        tup[index]['addressed'] = True
        new_state = board.change_board_state(index)
        is_terminal = not any(v['addressed'] is False for v in tup)
        return Prioritizer(tup, new_state, is_terminal)

    def change_board_state(board, index):
        td = board.tup[index]
        state = board.state

        state['lines_of_code'] += td['lines_changed']
        state['lines'] += td['lines_changed']
        state['new_lines'] += abs(td['lines_changed'])
        state['debt_maintain'] -= td['debt_maintain']

        if index == 3:
            state['bugs'] = 0
            state['rate_reliable'] = 5
            state['rem_eff_rel'] = 0
        return state

    def to_pretty_string(board):
        to_char = lambda v: ("X" if v['addressed'] is True else "O")
        rows = [[to_char(board.tup[col]) for col in range(4)], ]
        return (
                "\n  1 2 3 4\n"
                "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
                + "\n"
        )


def prioritizatize():
    tree = MCTS()
    board = prioritization_board()
    print(board.to_pretty_string())  # Prints TDs
    while True:
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break


def prioritization_board():
    tech_debts = ({
                      'addressed': False,
                      'spend': 1,
                      'defined': 5,
                      'lines_changed': 0,
                      'debt_maintain': 0,
                      'id': 1,
                  }, {
                      'addressed': False,
                      'spend': 1,
                      'defined': 1,
                      'lines_changed': -3,  # actually 9 lines
                      'debt_maintain': 1,
                      'id': 2,
                  }, {
                      'addressed': False,
                      'spend': float(1.5),
                      'defined': 2,
                      'lines_changed': 1,
                      'debt_maintain': 0,
                      'id': 3,
                  }, {
                      'addressed': False,
                      'spend': 2,
                      'defined': 5,
                      'lines_changed': 0,
                      'debt_maintain': 5,
                      'id': 4,
                  },
    )
    project_state = {
        'lines_of_code': 224, 'lines': 266,
        'statements': 85, 'functions': 12,
        'classes': 4, 'files': 8,
        'comments': 2,
        'cyclomatic': 19, 'cognitive': 11,
        'issues': 4,
        'dupl_lines': 0, 'dupl_blocks': 0,
        'debt_maintain': 6, 'rate_maintain': 5,
        'vulnerabilites': 0, 'rate_sec': 5,
        'rem_eff_sec': 0, 'bugs': 1,
        'rate_reliable': 3, 'rem_eff_rel': 2,
        'new_lines': 0,
    }
    return Prioritizer(tup=tech_debts, state=project_state, terminal=False)


if __name__ == "__main__":
    prioritizatize()
