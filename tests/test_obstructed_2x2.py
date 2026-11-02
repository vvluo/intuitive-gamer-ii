"""Smoke test: 2x2, k=2, where nature blocks one square at initialization.

The obstruction is nature's move, so the game opens on a chance node and the
first ply of every match belongs to CHANCE. Three squares then remain, and on
a 2x2 board *any two* of the surviving three form a line (row, column or a
diagonal), so the player to move first takes two of the three and always wins.
That gives the test a hard analytic answer to assert against.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import random, collections
from models.game import IPGame, P0, P1, CHANCE, Rewards
from models.simulator import UniformRandomPlayer, play_game, play_matches

BLOCKED = "#"
EMPTY = "_"
LINES = [((0,0),(0,1)), ((1,0),(1,1)),      # rows
         ((0,0),(1,0)), ((0,1),(1,1)),      # cols
         ((0,0),(1,1)), ((0,1),(1,0))]      # diagonals


class Obstructed2x2(IPGame[tuple[tuple[str, ...], int], tuple[int, int], tuple[str, ...]]):
    """2x2 two-in-a-row. Nature blocks one of the four squares before play.

    State is (flat 4-cell board, ply). Ply 0 is nature's; player moves start at
    ply 1, so the seat on move is (ply - 1) % 2.
    """

    marks = ("X", "O")

    def initial_state(self):
        return ((EMPTY,) * 4, 0)

    # -- nature ----------------------------------------------------------
    def is_chance_node(self, s):
        return s[1] == 0

    def chance_outcomes(self, s):
        return [((i // 2, i % 2), 0.25) for i in range(4)]

    # -- reading the state -----------------------------------------------
    def current_player(self, s):
        return CHANCE if s[1] == 0 else (s[1] - 1) % 2

    def information_set(self, s, player):
        return s[0]

    def legal_moves(self, s, player):
        if self.is_chance_node(s):
            return [a for a, _ in self.chance_outcomes(s)]
        return [(i // 2, i % 2) for i, c in enumerate(s[0]) if c == EMPTY]

    def is_terminal(self, s):
        if s[1] == 0:
            return False
        return self._winner(s[0]) is not None or EMPTY not in s[0]

    # -- dynamics ---------------------------------------------------------
    def apply_action(self, s, a):
        board, ply = s
        cells = list(board)
        cells[a[0] * 2 + a[1]] = BLOCKED if ply == 0 else self.marks[(ply - 1) % 2]
        return (tuple(cells), ply + 1)

    def reward(self, s, a, ns) -> Rewards:
        w = self._winner(ns[0])
        if w is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if w == P0 else (-1.0, 1.0)

    def _winner(self, board):
        for (r1, c1), (r2, c2) in LINES:
            a, b = board[r1 * 2 + c1], board[r2 * 2 + c2]
            if a == b and a in self.marks:
                return self.marks.index(a)
        return None


ok = lambda label, cond: print(f"  {'PASS' if cond else 'FAIL'}  {label}")

g = Obstructed2x2()
N = 400
res = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], N,
                   rng=random.Random(11))

tally = collections.Counter(r.winner for r in res)
blocked = collections.Counter(
    r.moves[0].action for r in res
)
print(f"  outcomes: P0={tally[P0]} P1={tally[P1]} draw={tally[None]}")
print(f"  obstruction square counts: {dict(sorted(blocked.items()))}")
print(f"  example final board: {res[0].final_state[0]}")

print("\nchance / initialization")
ok("every game opens on a CHANCE ply", all(r.moves[0].player == CHANCE for r in res))
ok("nature moved exactly once per game",
   all(sum(1 for m in r.moves if m.player == CHANCE) == 1 for r in res))
ok("all four squares get blocked across games", len(blocked) == 4)
ok("obstruction roughly uniform (70..130 each)",
   all(70 <= v <= 130 for v in blocked.values()))
ok("exactly one blocked cell on every final board",
   all(r.final_state[0].count(BLOCKED) == 1 for r in res))
ok("nature's square is the one blocked",
   all(r.final_state[0][m.action[0]*2 + m.action[1]] == BLOCKED
       for r in res for m in r.moves if m.player == CHANCE))

print("\nplay")
ok("no player ever moved into the blocked square",
   all(len({m.action for m in r.moves}) == len(r.moves) for r in res))
ok("every game is 4 plies (1 chance + 3 moves)", all(r.length == 4 for r in res))
ok("board full at the end", all(EMPTY not in r.final_state[0] for r in res))
ok("P0 wins every game (analytic: any 2 of 3 survivors form a line)",
   tally[P0] == N)
ok("zero-sum returns", all(r.returns[0] + r.returns[1] == 0 for r in res))
ok("returns are exactly (1,-1)", all(r.returns == (1.0, -1.0) for r in res))
ok("reward is 0 until the final transition",
   all(all(m.rewards == (0.0, 0.0) for m in r.moves[:-1]) for r in res))

print("\nseating")
alt = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 20,
                   rng=random.Random(5), alternate_seats=True)
ok("seat P0 still wins every game under alternation",
   all(r.winner == P0 for r in alt))
ok("so the advantage is the seat's, not the policy's",
   {r.seating for r in alt} == {(P0, P1), (P1, P0)})

print("\nreproducibility")
a = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 30, rng=random.Random(99))
b = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 30, rng=random.Random(99))
ok("same seed reproduces nature's draws too",
   [m.action for r in a for m in r.moves] == [m.action for r in b for m in r.moves])
