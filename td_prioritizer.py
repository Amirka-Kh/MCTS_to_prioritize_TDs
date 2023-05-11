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
from constants import TD_TEMPLATE, BEGIN_STATE

_TTTB = namedtuple("Prioritizer", "tup state terminal")

TD = namedtuple('TD', TD_TEMPLATE)
State = namedtuple('State', BEGIN_STATE)


# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Prioritizer(_TTTB, Node):
    def find_children(board):
        if board.terminal:  # If no more refactorings left this node\state is terminal
            return set()
        # Otherwise, you can refactor available TDs
        return {
            board.make_move(i) for i, value in enumerate(board.tup) if value.addressed is False
        }

    def find_random_child(board):
        if board.terminal:  # If no more refactorings left this node\state is terminal
            return None
        empty_spots = [i for i, value in enumerate(board.tup) if value.addressed is False]
        return board.make_move(choice(empty_spots))

    def reward(board):
        if not board.terminal:
            raise RuntimeError(f"reward called on non-terminal board {board}")
        st = board.state

        reward = st.lines / (st.statements + st.functions + st.classes + st.files + st.comments)
        reward += (4 - st.issues) / 4 + (st.debt_maintain + st.rem_eff_rel) / 13
        return reward

    def is_terminal(board):
        return board.terminal  # returns boolean

    def make_move(board, index):
        tds_list = []
        for i, value in enumerate(board.tup):
            if i == index:
                new_td = TD(
                    True, value.spend, value.defined, value.lines_changed,
                    value.debt_maintain, value.id,
                )
                tds_list.append(new_td)
                continue
            tds_list.append(value)

        new_board_tup = tuple(tds_list)
        new_state = board.change_board_state(index)
        is_terminal = not any(v.addressed is False for v in new_board_tup)
        return Prioritizer(new_board_tup, new_state, is_terminal)

    def change_board_state(board, index):
        td = board.tup[index]
        state = board.state

        lines_of_code = state.lines_of_code + td.lines_changed
        lines = state.lines + td.lines_changed
        new_lines = state.new_lines + abs(td.lines_changed)
        debt_m = state.debt_maintain - td.debt_maintain
        bugs = state.bugs
        rate_reliable = state.rate_reliable
        rem_eff_rel = state.rem_eff_rel
        if index == 3:
            bugs = 0
            rate_reliable = 5
            rem_eff_rel = 0
        return State(
            lines_of_code, lines, state.statements, state.functions, state.classes, state.files,
            state.comments, state.cyclomatic, state.cognitive, state.issues, state.dupl_lines,
            state.dupl_blocks, debt_m, state.rate_maintain, state.vulnerabilites,
            state.rate_sec, state.rem_eff_sec, bugs, rate_reliable, rem_eff_rel, new_lines,
        )

    def to_pretty_string(board):
        to_char = lambda v: ("yes" if v.addressed is True else "___")
        row = [to_char(board.tup[col]) for col in range(4)]
        return "TD1 TD2 TD3 TD4\n" + " ".join(row)


def prioritizatize():
    tree = MCTS()
    board = prioritization_board()
    print(board.to_pretty_string())
    while True:
        # train tree for several iteration to find the best possible moves
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)  # refactoring chosen, TD eliminated\addressed
        print(board.to_pretty_string())
        if board.terminal:
            break


def prioritization_board():
    tech_debts = (
        TD(False, 1, 5, 0, 0, 1),
        TD(False, 1, 1, -3, 1, 2),
        TD(False, 1.5, 2, 1, 0, 3),
        TD(False, 2, 5, 0, 5, 4),
    )
    initial_state = State(
        lines_of_code=224, lines=266, statements=85, functions=12, classes=4, files=8, comments=2, cyclomatic=19,
        cognitive=11, issues=4, dupl_lines=0, dupl_blocks=0, debt_maintain=6, rate_maintain=5, vulnerabilites=0,
        rate_sec=5, rem_eff_sec=0, bugs=1, rate_reliable=3, rem_eff_rel=2, new_lines=0,
    )
    return Prioritizer(tup=tech_debts, state=initial_state, terminal=False)


if __name__ == "__main__":
    prioritizatize()
