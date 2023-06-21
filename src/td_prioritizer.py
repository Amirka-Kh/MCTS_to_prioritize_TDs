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
from src.monte_carlo_tree_search import MCTS, Node
from src.constants import TD_TEMPLATE, BEGIN_STATE
from data.tests import SMALL, MEDIUM, BIG, STP, MTP, BTP
import matplotlib.pyplot as plt

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
        st = board.state
        tup = board.tup
        last_tup = [i for i in tup if i.last is True][0]
        overall_time = sum([i.spend for i in tup])

        cost = last_tup.spend / overall_time + last_tup.lines_changed / st.lines

        reward = (st.statements + st.functions + st.classes + st.files + st.comments) / st.lines + \
                 (len(tup) - st.issues) / len(tup) + \
                 last_tup.debt_maintain / (last_tup.debt_maintain + st.debt_maintain + 0.01) + \
                 last_tup.remediation_time / (last_tup.remediation_time + st.rem_eff_rel + 0.01)
        # last_tup.debt_maintain / sum([i.debt_maintain for i in tup]) + \
        # last_tup.remediation_time / sum([i.remediation_time for i in tup])
        return reward-0.7*cost

    def is_terminal(board):
        return board.terminal  # returns boolean

    def make_move(board, index):
        tds_list = []
        for i, value in enumerate(board.tup):
            if i == index:
                new_td = TD(
                    True, value.spend, value.defined, value.lines_changed,
                    value.debt_maintain, value.remediation_time, True, value.id,
                )
                tds_list.append(new_td)
                continue
            tds_list.append(TD(
                value.addressed, value.spend, value.defined, value.lines_changed,
                value.debt_maintain, value.remediation_time, False, value.id,
            ))
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
        rem_eff_rel = state.rem_eff_rel - td.remediation_time
        if rem_eff_rel == 0:
            rate_reliable, bugs = 5, 0
        return State(
            lines_of_code, lines, state.statements, state.functions, state.classes, state.files,
            state.comments, state.cyclomatic, state.cognitive, state.issues, state.dupl_lines,
            state.dupl_blocks, debt_m, state.rate_maintain, state.vulnerabilites,
            state.rate_sec, state.rem_eff_sec, bugs, rate_reliable, rem_eff_rel, new_lines,
        )


def prioritizatize():
    max_sim = 30
    stats = []
    test_project_size = 'default'
    for sim_num in range(0, max_sim):
        positions = []
        board = prioritization_board(test_project_size)
        tree = MCTS()
        tree._expand(board)
        while True:
            # train tree for several iteration to find the best possible moves
            for _ in range(sim_num):
                tree.do_rollout(board)
            board = tree.choose(board)  # refactoring chosen, TD eliminated\addressed
            for td in board.tup:
                if td.id in positions:
                    continue
                if td.addressed:
                    positions.append(int(td.id))
            if board.terminal:
                stats.append(positions)
                break
    if test_project_size == 'default':
        first = [i[0] for i in stats]
        second = [i[1] for i in stats]
        third = [i[2] for i in stats]
        forth = [i[3] for i in stats]
        # fifth = [i[4] for i in stats]
        plt.title('TD prioritization on rollouts count', fontsize=20, fontname='Times New Roman')
        plt.ylabel('TD number', color='gray')
        plt.xlabel('Simulations count', color='gray')
        plt.grid(True)
        # plt.plot([i for i in range(0, max_sim)], first, 'b', second, 'g', third, 'r', forth, 'c', fifth, 'y', linewidth=2.0)
        plt.plot([i for i in range(0, max_sim)], first, 'b', second, 'g', third, 'r', forth, 'c', linewidth=2.0)
        plt.legend(['First', 'Second', 'Third', 'Forth', 'Fifth'], loc=4)
        plt.show()


def prioritization_board(size):
    tech_debts, initial_state = None, None

    if size == 'small':
        tmp = []
        for i in SMALL:
            tmp.append(TD._make(i))
        tech_debts = tuple(tmp)
        initial_state = State._make(STP)
    elif size == 'medium':
        tmp = []
        for i in MEDIUM:
            tmp.append(TD._make(i))
        tech_debts = tuple(tmp)
        initial_state = State._make(MTP)
    elif size == 'big':
        tmp = []
        for i in BIG:
            tmp.append(TD._make(i))
        tech_debts = tuple(tmp)
        initial_state = State._make(BTP)
    else:
        tech_debts = (
            TD(False, 1, 5, 0, 0, 0, False, 1),
            TD(False, 1, 1, -3, 1, 0, False, 2),
            TD(False, 1.5, 2, 1, 0, 2, False, 3),
            TD(False, 2, 5, 0, 5, 0, False, 4),
        )
        initial_state = State(
            lines_of_code=224, lines=266, statements=85, functions=12, classes=4, files=8, comments=2, cyclomatic=19,
            cognitive=11, issues=4, dupl_lines=0, dupl_blocks=0, debt_maintain=6, rate_maintain=5, vulnerabilites=0,
            rate_sec=5, rem_eff_sec=0, bugs=1, rate_reliable=3, rem_eff_rel=2, new_lines=0,
        )

    return Prioritizer(tup=tech_debts, state=initial_state, terminal=False)


if __name__ == "__main__":
    prioritizatize()
