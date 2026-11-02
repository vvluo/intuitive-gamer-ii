"""Abstract game interface.

`IPGame` describes a finite, turn-taking game played by two competing players.
"IP" is for *imperfect information*: a player never receives the state, only the
information set the state puts them in. For perfect-information games such as
tic-tac-toe the information set is simply the position, and `information_set`
is close to the identity -- but routing every observation through it means a
partially-observable variant needs no change to the player API.

Design commitments worth stating, since they are easy to erode later:

* **State is a value, not a place.** Every method takes the state explicitly and
  `apply_action` returns a *new* state rather than mutating its argument. The
  game object itself holds only the rules.
* **One source of truth for whose turn it is.** `current_player(state)` is that
  source. No method takes both a state and a "player to move" that could
  disagree with it; where a method does take `player`, it is asking a
  counterfactual question about a player who may not be on move.
* **Rewards are per-player and tuple-valued.** A win/loss game returns
  `(0.0, 0.0)` on every transition except the last. A game with shaped or
  per-action payoffs returns whatever it likes on each. Both fit the same
  signature, so the simulator never needs to know which kind it is driving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, Sequence, TypeVar

__all__ = [
    "StateT",
    "ActionT",
    "InfoSetT",
    "Rewards",
    "P0",
    "P1",
    "CHANCE",
    "Transition",
    "IPGame",
]

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
InfoSetT = TypeVar("InfoSetT")

#: Per-player payoff, indexed by seat.
Rewards = tuple[float, float]

#: Seats. Players are indexed from 0 so they can index into tuples directly.
P0: int = 0
P1: int = 1

#: Pseudo-seat returned by `current_player` at a chance node. Negative so it can
#: never be mistaken for a real seat index.
CHANCE: int = -1


@dataclass(frozen=True)
class Transition(Generic[StateT, ActionT]):
    """One applied action and everything that followed from it."""

    state: StateT
    player: int
    action: ActionT
    next_state: StateT
    rewards: Rewards
    terminal: bool


class IPGame(ABC, Generic[StateT, ActionT, InfoSetT]):
    """A two-player competitive game.

    Subclasses implement seven methods. Everything else here is composed from
    them and may be overridden for efficiency, but need not be.
    """

    #: Fixed at two for now. Chance, when enabled, is not a player.
    num_players: ClassVar[int] = 2

    #: Whether the two players' payoffs are constrained to sum to zero. Left as
    #: an advertisement rather than an enforcement; the simulator reads it to
    #: decide whether it may infer a winner from the returns.
    zero_sum: ClassVar[bool] = True

    # -- initialization ---------------------------------------------------

    @abstractmethod
    def initial_state(self) -> StateT:
        """Return the state the game starts in."""

    # -- reading the state ------------------------------------------------

    @abstractmethod
    def current_player(self, state: StateT) -> int:
        """Return the seat to move in `state`.

        Returns `CHANCE` if the next move belongs to nature. Behaviour at a
        terminal state is undefined; call `is_terminal` first.
        """

    @abstractmethod
    def information_set(self, state: StateT, player: int) -> InfoSetT:
        """Return everything `player` is entitled to know about `state`.

        This is the only channel by which a player learns anything, so it must
        not leak private information belonging to the opponent. It takes an
        arbitrary `player` rather than assuming the one on move, so that a
        player reasoning about its opponent can ask what they can see.

        The return value should be hashable and treated as immutable by
        callers; players are permitted to use it as a dictionary key.
        """

    @abstractmethod
    def legal_moves(self, state: StateT, player: int) -> Sequence[ActionT]:
        """Return the actions available to `player` in `state`.

        Like `information_set` this accepts any seat, not only the one on move:
        asking what the opponent *could* do is how a heuristic evaluates a
        threat without searching a reply. Returns an empty sequence when the
        player has no move.
        """

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool:
        """Return whether `state` ends the game."""

    # -- dynamics ---------------------------------------------------------

    @abstractmethod
    def apply_action(self, state: StateT, action: ActionT) -> StateT:
        """Return the state reached by playing `action` in `state`.

        Must not mutate `state`. The acting seat is `current_player(state)` and
        is deliberately not a parameter.
        """

    @abstractmethod
    def reward(
        self, state: StateT, action: ActionT, next_state: StateT
    ) -> Rewards:
        """Return the per-seat payoff for the transition.

        For a plain win/loss game this is `(0.0, 0.0)` everywhere except the
        transition into a terminal state. For a game with per-action costs or
        shaping it may be nonzero throughout. Indexed by seat, so
        `reward(...)[P0]` is what player 0 collected.
        """

    # -- composed ---------------------------------------------------------

    def step(
        self, state: StateT, action: ActionT, *, validate: bool = True
    ) -> Transition[StateT, ActionT]:
        """Apply `action` and package the result.

        This is the single entry point the simulator uses to advance a game.
        Set `validate=False` to skip the legality check in hot loops where the
        action is known to have come from `legal_moves`.
        """
        if validate and self.is_terminal(state):
            raise ValueError("cannot act in a terminal state")

        player = self.current_player(state)

        if validate and action not in self.legal_moves(state, player):
            raise ValueError(f"illegal action {action!r} for player {player}")

        next_state = self.apply_action(state, action)
        return Transition(
            state=state,
            player=player,
            action=action,
            next_state=next_state,
            rewards=self.reward(state, action, next_state),
            terminal=self.is_terminal(next_state),
        )

    def opponent(self, player: int) -> int:
        """Return the other seat. Only meaningful for real seats."""
        if player not in (P0, P1):
            raise ValueError(f"{player} is not a player seat")
        return P1 if player == P0 else P0

    # -- chance -----------------------------------------------------------
    #
    # Inert by default: with `is_chance_node` returning False the rest of the
    # framework behaves as a deterministic two-player game. A subclass that
    # wants nature overrides both methods and starts returning CHANCE from
    # `current_player`; nothing else has to change.

    def is_chance_node(self, state: StateT) -> bool:
        """Return whether nature, rather than a player, moves in `state`."""
        return False

    def chance_outcomes(
        self, state: StateT
    ) -> Sequence[tuple[ActionT, float]]:
        """Return `(action, probability)` pairs for nature's move.

        Probabilities should sum to one. Only called where `is_chance_node` is
        true, so the default need not be useful.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no chance nodes; "
            "override is_chance_node and chance_outcomes to add them"
        )
