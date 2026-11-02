"""Player abstraction and the player-versus-player simulator.

A `Player` is a policy. It is handed an information set and the moves available
in it, and returns one of them. It is never handed the state, so a player
physically cannot condition on information the game did not disclose -- the
information barrier is enforced by the signature rather than by convention.

Game rules are common knowledge, so a player *is* given the game object once,
in `reset`, before play begins. That is where a policy that needs the rules
(how many in a row wins, which directions count) should cache them. Reaching
through that reference to read live state during `act` defeats the point.

`play_game` drives one match; `play_matches` runs a batch and can alternate
seats, which matters because several game variants hand one seat an explicit
advantage.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Sequence

from models.game import (
    CHANCE,
    P0,
    P1,
    ActionT,
    IPGame,
    InfoSetT,
    Rewards,
    StateT,
    Transition,
)

__all__ = [
    "Player",
    "UniformRandomPlayer",
    "MoveRecord",
    "GameResult",
    "play_game",
    "play_matches",
]


# ---------------------------------------------------------------------------
# The player abstraction
# ---------------------------------------------------------------------------


class Player(ABC, Generic[InfoSetT, ActionT]):
    """A policy over information sets.

    Only `act` is required. The hooks exist so that stateful policies -- ones
    that cache rules, carry a search tree between moves, or learn from what
    they see -- can be dropped in without changing the simulator.
    """

    #: Short identifier used when labelling results.
    name: str = "player"

    def reset(
        self,
        game: IPGame[StateT, ActionT, InfoSetT],
        seat: int,
        rng: random.Random | None = None,
    ) -> None:
        """Prepare for a new game in which this policy occupies `seat`.

        Called once before the first move. `game` is supplied because the rules
        are common knowledge; cache what is needed from it here. `rng`, when
        given, is the simulator's generator -- adopting it is what makes a run
        reproducible from a single seed.
        """

    @abstractmethod
    def act(
        self, info_set: InfoSetT, legal_moves: Sequence[ActionT]
    ) -> ActionT:
        """Choose one of `legal_moves`.

        `legal_moves` is guaranteed non-empty. The return value must be an
        element of it.
        """

    def observe(self, transition: Transition[StateT, ActionT]) -> None:
        """Note a transition that has just occurred.

        Called for both players' moves, so this is also how a policy watches
        its opponent. The transition carries the full state, which a policy
        under genuine imperfect information must not read; it is passed whole
        because the simulator cannot know which projection a given policy is
        entitled to. Default is to ignore it.
        """

    def observe_result(self, returns: Rewards) -> None:
        """Note the final per-seat returns. Default is to ignore them."""


class UniformRandomPlayer(Player[InfoSetT, ActionT]):
    """Picks uniformly at random among the legal moves.

    The floor baseline: it reads neither the information set nor the rules, so
    every variant of a game looks identical to it.
    """

    name = "uniform_random"

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def reset(
        self,
        game: IPGame[StateT, ActionT, InfoSetT],
        seat: int,
        rng: random.Random | None = None,
    ) -> None:
        if rng is not None:
            self._rng = rng

    def act(
        self, info_set: InfoSetT, legal_moves: Sequence[ActionT]
    ) -> ActionT:
        return self._rng.choice(list(legal_moves))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveRecord(Generic[ActionT]):
    """One ply of a completed game."""

    ply: int
    player: int  # a seat, or CHANCE
    action: ActionT
    rewards: Rewards


@dataclass
class GameResult(Generic[StateT, ActionT]):
    """The outcome of one match."""

    moves: list[MoveRecord[ActionT]] = field(default_factory=list)
    returns: Rewards = (0.0, 0.0)
    final_state: StateT | None = None
    #: True if play stopped at `max_plies` rather than at a terminal state.
    truncated: bool = False
    #: Seat occupied by each player, in the order they were passed in. With
    #: `alternate_seats` this is how you tell which way round a match was.
    seating: tuple[int, int] = (P0, P1)

    @property
    def length(self) -> int:
        """Number of plies played, chance moves included."""
        return len(self.moves)

    @property
    def winner(self) -> int | None:
        """Seat with the strictly greater return, or None for a draw.

        Only meaningful for a zero-sum game that ran to termination.
        """
        if self.truncated:
            return None
        if self.returns[P0] > self.returns[P1]:
            return P0
        if self.returns[P1] > self.returns[P0]:
            return P1
        return None


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


def play_game(
    game: IPGame[StateT, ActionT, InfoSetT],
    players: Sequence[Player[InfoSetT, ActionT]],
    *,
    rng: random.Random | None = None,
    max_plies: int | None = None,
    seating: tuple[int, int] = (P0, P1),
) -> GameResult[StateT, ActionT]:
    """Play one match and return the result.

    `players[i]` occupies `seating[i]`, so passing `seating=(P1, P0)` swaps
    sides without reordering the player objects. `rng` seeds both nature and
    any player that adopts it in `reset`; pass one to make a run reproducible.
    `max_plies` guards against a policy that will not terminate, and marks the
    result `truncated` if it bites.
    """
    if len(players) != game.num_players:
        raise ValueError(
            f"{game.num_players} players expected, got {len(players)}"
        )
    if set(seating) != {P0, P1}:
        raise ValueError(f"seating must be a permutation of seats, got {seating}")

    rng = rng if rng is not None else random.Random()

    # seat -> policy occupying it
    by_seat: dict[int, Player[InfoSetT, ActionT]] = {
        seat: player for seat, player in zip(seating, players)
    }
    for seat, player in by_seat.items():
        player.reset(game, seat, rng)

    state = game.initial_state()
    result: GameResult[StateT, ActionT] = GameResult(seating=seating)
    totals = [0.0, 0.0]
    ply = 0

    while not game.is_terminal(state):
        if max_plies is not None and ply >= max_plies:
            result.truncated = True
            break

        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            actions = [a for a, _ in outcomes]
            weights = [p for _, p in outcomes]
            action = rng.choices(actions, weights=weights)[0]
            mover = CHANCE
        else:
            mover = game.current_player(state)
            info_set = game.information_set(state, mover)
            legal = game.legal_moves(state, mover)
            if not legal:
                raise ValueError(
                    f"player {mover} has no legal moves in a non-terminal state"
                )
            action = by_seat[mover].act(info_set, legal)
            if action not in legal:
                raise ValueError(
                    f"{by_seat[mover].name} returned illegal action {action!r}"
                )

        transition = game.step(state, action, validate=False)

        for seat in (P0, P1):
            totals[seat] += transition.rewards[seat]
        result.moves.append(
            MoveRecord(
                ply=ply,
                player=mover,
                action=action,
                rewards=transition.rewards,
            )
        )

        for player in by_seat.values():
            player.observe(transition)

        state = transition.next_state
        ply += 1

    result.final_state = state
    result.returns = (totals[P0], totals[P1])
    for player in by_seat.values():
        player.observe_result(result.returns)
    return result


def play_matches(
    game: IPGame[StateT, ActionT, InfoSetT],
    players: Sequence[Player[InfoSetT, ActionT]],
    num_games: int,
    *,
    rng: random.Random | None = None,
    max_plies: int | None = None,
    alternate_seats: bool = False,
) -> list[GameResult[StateT, ActionT]]:
    """Play `num_games` matches and return every result.

    With `alternate_seats`, `players[0]` takes seat 0 on even-numbered games
    and seat 1 on odd-numbered ones. Several variants give one seat a standing
    advantage, so a matchup reported from one seating alone is not a comparison
    between the two policies.
    """
    rng = rng if rng is not None else random.Random()
    results = []
    for i in range(num_games):
        seating = (P1, P0) if (alternate_seats and i % 2) else (P0, P1)
        results.append(
            play_game(
                game,
                players,
                rng=rng,
                max_plies=max_plies,
                seating=seating,
            )
        )
    return results
