from typing import List, Dict, Tuple, Callable
import json
import inspect


class GameTheoryGame:
    """
    A game theoretic game

    Attributes:
        n_iterations: Number of iterations
        memory_size: Size of history available
    """

    def __init__(
        self,
        n_iterations: int = 1,
        memory_size: int = 1,
        store_stats: bool = False,
        out_file_name: str = "",
    ) -> None:
        """ Constructor
        """
        self.n_iterations = n_iterations
        self.memory_size = memory_size
        self.store_stats = store_stats
        self.out_file_name = out_file_name

        if self.store_stats:
            with open(self.out_file_name, "w") as out_file:
                json.dump([], out_file)

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        raise NotImplementedError("Implement in game")

    @staticmethod
    def get_move(
        player: Callable[[List[Tuple[str, str]], int], str], history: List[str], iteration: int
    ) -> str:
        """ Helper function to get the player move.

        Player is a function that takes the history and current iteration into account
        """
        move = player(history, iteration)
        return move

    def run(
        self,
        player_1: Callable[[List[Tuple[str, str]], int], str],
        player_2: Callable[[List[Tuple[str, str]], int], str],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[str, str]]]:
        """Return the payoff for each iteration of the game.
        """
        history: Dict[str, List[str]] = {
            "player_1": [""] * self.memory_size,
            "player_2": [""] * self.memory_size,
        }
        payoffs: List[Tuple[float, float]] = []
        _payoff = self.get_payoff()
        for i in range(self.n_iterations):
            move_1 = GameTheoryGame.get_move(player_1, history["player_2"], i)
            history["player_1"].append(move_1)
            move_2 = GameTheoryGame.get_move(player_2, history["player_1"], i)
            history["player_2"].append(move_2)
            moves = (move_1, move_2)
            payoffs.append(_payoff[moves])

        if self.store_stats:
            self.dump_stats(player_1, player_2, payoffs, history)

        return payoffs, history

    def revise_history(self, history: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        revised_history: List[Tuple[str, str]] = []
        for i in range(self.memory_size, len(history["player_1"])):
            revised_history.append((history["player_1"][i], history["player_2"][i]))

        return revised_history

    def dump_stats(
        self,
        player_1: Callable[[List[Tuple[str, str]], int], str],
        player_2: Callable[[List[Tuple[str, str]], int], str],
        payoffs: List[Tuple[float, float]],
        history: Dict[str, List[str]],
    ) -> None:
        """ Append run statistics to JSON file.

        Note, File IO can be slow.
        """
        revised_history: List[Tuple[str, str]] = self.revise_history(history)
        data = {
            "player_1": str(inspect.getsourcelines(player_1)[0]),
            "player_2": str(inspect.getsourcelines(player_2)[0]),
            "payoffs": payoffs,
            "history": revised_history,
        }
        with open(self.out_file_name, "r") as in_file:
            json_data = json.load(in_file)

        json_data.append(data)
        with open(self.out_file_name, "w") as out_file:
            json.dump(json_data, out_file)


class PrisonersDilemma(GameTheoryGame):
    """
    Prisoners Dilemma game, see https://en.wikipedia.org/wiki/Prisoner%27s_dilemma

    """

    COOPERATE: str = "C"
    DEFECT: str = "D"
    R: float = 1.0  # Reward
    P: float = 2.0  # Penalty
    S: float = 3.0  # Sucker
    T: float = 0.0  # Temptation
    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (COOPERATE, COOPERATE): (R, R),
        (COOPERATE, DEFECT): (S, T),
        (DEFECT, COOPERATE): (T, S),
        (DEFECT, DEFECT): (P, P),
    }
    DEFAULT_OUT_FILE: str = "ipd_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return PrisonersDilemma.PAYOFF


# TODO  Implement game from 'Ecotypic variation in the asymmetric Hawk-Dove game: when is Bourgeois an evolutionarily stable strategy?', Michael Mesterton-Gibbons
class HawkAndDove(GameTheoryGame):
    """
    Hawk And Dove game, see https://en.wikipedia.org/wiki/Chicken_(game)

    """

    HAWK: str = "H"
    DOVE: str = "D"
    V: float = 2.0
    C: float = 4.0
    PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
        (HAWK, HAWK): ((V - C) / 2.0, (V - C) / 2.0),
        (HAWK, DOVE): (V, 0),
        (DOVE, HAWK): (0, V),
        (DOVE, DOVE): (V / 2.0, V / 2.0),
    }
    DEFAULT_OUT_FILE: str = "had_stats.json"

    def get_payoff(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Return payoff for each strategy combination."""
        return HawkAndDove.PAYOFF
