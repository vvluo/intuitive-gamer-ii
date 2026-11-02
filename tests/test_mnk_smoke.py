"""Throwaway smoke test: a concrete m,n,k game + chance game against the ABCs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import random, collections
from models.game import IPGame, P0, P1, CHANCE, Rewards
from models.simulator import Player, UniformRandomPlayer, play_game, play_matches

Board = tuple[str, ...]  # flat, row-major; "_" empty

class MNK(IPGame[tuple[Board, int], int, Board]):
    """k-in-a-row on an m x n board. State = (board, ply)."""
    def __init__(self, m=3, n=3, k=3):
        self.m, self.n, self.k = m, n, k
        self.marks = ("X", "O")

    def initial_state(self): return (tuple("_" * (self.m * self.n)), 0)
    def current_player(self, s): return s[1] % 2
    def information_set(self, s, player): return s[0]          # perfect info
    def legal_moves(self, s, player):
        return [i for i, c in enumerate(s[0]) if c == "_"]
    def apply_action(self, s, a):
        board, ply = s
        new = list(board); new[a] = self.marks[ply % 2]
        return (tuple(new), ply + 1)
    def is_terminal(self, s):
        return self._winner(s[0]) is not None or "_" not in s[0]
    def reward(self, s, a, ns) -> Rewards:
        w = self._winner(ns[0])
        if w is None: return (0.0, 0.0)
        return (1.0, -1.0) if w == P0 else (-1.0, 1.0)
    def _winner(self, board):
        m, n, k = self.m, self.n, self.k
        g = lambda r, c: board[r * n + c]
        for r in range(m):
            for c in range(n):
                if g(r, c) == "_": continue
                for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
                    er, ec = r + (k-1)*dr, c + (k-1)*dc
                    if not (0 <= er < m and 0 <= ec < n): continue
                    if all(g(r+i*dr, c+i*dc) == g(r,c) for i in range(k)):
                        return self.marks.index(g(r, c))
        return None

class CoinThenPick(IPGame[tuple[int, int], int, tuple[int, int]]):
    """Nature flips first; the winner of the flip takes the point. Tests chance."""
    def initial_state(self): return (-1, 0)          # (flip result, ply)
    def is_chance_node(self, s): return s[0] == -1
    def chance_outcomes(self, s): return [(0, 0.5), (1, 0.5)]
    def current_player(self, s): return CHANCE if s[0] == -1 else P0
    def information_set(self, s, player): return s
    def legal_moves(self, s, player): return [0] if s[0] != -1 else [0, 1]
    def apply_action(self, s, a): return (a, s[1] + 1) if s[0] == -1 else (s[0], s[1] + 1)
    def is_terminal(self, s): return s[1] >= 2
    def reward(self, s, a, ns):
        if not self.is_terminal(ns): return (0.0, 0.0)
        return (1.0, -1.0) if ns[0] == 0 else (-1.0, 1.0)

ok = lambda label, cond: print(f"  {'PASS' if cond else 'FAIL'}  {label}")

print("1. 3x3 random vs random, 400 games")
g = MNK(3, 3, 3)
res = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 400, rng=random.Random(0))
tally = collections.Counter(r.winner for r in res)
print(f"     P0={tally[P0]} P1={tally[P1]} draw={tally[None]}  "
      f"mean_len={sum(r.length for r in res)/len(res):.2f}")
ok("all games terminal", all(not r.truncated for r in res))
ok("lengths in 5..9", all(5 <= r.length <= 9 for r in res))
ok("zero-sum returns", all(r.returns[0] + r.returns[1] == 0 for r in res))
ok("P0 wins most (known ~58/29/13)", tally[P0] > tally[P1] > tally[None])

print("2. reproducibility")
a = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 50, rng=random.Random(7))
b = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 50, rng=random.Random(7))
ok("same seed -> same moves", [m.action for r in a for m in r.moves] ==
                              [m.action for r in b for m in r.moves])

print("3. seat alternation")
alt = play_matches(g, [UniformRandomPlayer(), UniformRandomPlayer()], 10,
                   rng=random.Random(1), alternate_seats=True)
ok("seatings alternate", [r.seating for r in alt][:4] == [(P0,P1),(P1,P0),(P0,P1),(P1,P0)])

print("4. larger / rectangular board")
g2 = MNK(4, 6, 4)
r2 = play_matches(g2, [UniformRandomPlayer(), UniformRandomPlayer()], 50, rng=random.Random(2))
ok("4x6 k=4 runs", all(not r.truncated for r in r2) and all(r.length <= 24 for r in r2))

print("5. chance nodes")
gc = CoinThenPick()
rc = play_matches(gc, [UniformRandomPlayer(), UniformRandomPlayer()], 200, rng=random.Random(3))
heads = sum(1 for r in rc if r.winner == P0)
ok("chance node fired", all(r.moves[0].player == CHANCE for r in rc))
ok(f"coin ~fair ({heads}/200)", 70 < heads < 130)

print("6. guards")
class Cheat(UniformRandomPlayer):
    def act(self, info_set, legal_moves): return 999
try:
    play_game(g, [Cheat(), UniformRandomPlayer()]); ok("illegal action raises", False)
except ValueError as e: ok(f"illegal action raises ({type(e).__name__})", True)
try:
    IPGame(); ok("ABC not instantiable", False)
except TypeError: ok("ABC not instantiable", True)
try:
    play_game(g, [UniformRandomPlayer()]); ok("wrong player count raises", False)
except ValueError: ok("wrong player count raises", True)
st = g.initial_state()
try:
    g.step(st, 999); ok("step validates legality", False)
except ValueError: ok("step validates legality", True)
ok("apply_action does not mutate", (g.apply_action(st, 0), st[0].count("_"))[1] == 9)

print("7. max_plies truncation")
t = play_game(g, [UniformRandomPlayer(), UniformRandomPlayer()], max_plies=3)
ok("truncated flag + winner None", t.truncated and t.winner is None and t.length == 3)
