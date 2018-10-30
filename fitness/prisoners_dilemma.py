from typing import List, Dict, Tuple, Callable
import json
import inspect


class PrisonersDilemma:
    """
    Prisoners Dilemma game

    Attributes:
        n_iterations: Number of iterations
        memory_size: Size of history available
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

    def __init__(
        self,
        n_iterations: int = 1,
        memory_size: int = 1,
        store_stats: bool = False,
        out_file_name: str = DEFAULT_OUT_FILE,
    ) -> None:
        """ Constructor
        """
        self.n_iterations = n_iterations
        self.memory_size = memory_size
        self.store_stats = store_stats
        self.out_file_name = out_file_name

        with open(self.out_file_name, "w") as out_file:
            json.dump([], out_file)

    @staticmethod
    def get_move(
        player: Callable[[List[Tuple[str, str]], int], str],
        history: List[Tuple[str, str]],
        iteration: int,
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
        """Return the sentence for each iteration of the game.
        """
        history: List[Tuple[str, str]] = [("", "")] * self.memory_size
        sentences: List[Tuple[float, float]] = []
        for i in range(self.n_iterations):
            move_1 = PrisonersDilemma.get_move(player_1, history, i)
            move_2 = PrisonersDilemma.get_move(player_2, history, i)
            moves = (move_1, move_2)
            sentences.append(PrisonersDilemma.PAYOFF[moves])
            history.append(moves)

        if self.store_stats:
            self.dump_stats(player_1, player_2, sentences, history)

        return sentences, history

    def dump_stats(
        self,
        player_1: Callable[[List[Tuple[str, str]], int], str],
        player_2: Callable[[List[Tuple[str, str]], int], str],
        sentences: List[Tuple[float, float]],
        history: List[Tuple[str, str]],
    ) -> None:
        """ Append run statistics to JSON file.

        Note, File IO can be slow.
        """
        data = {
            "player_1": str(inspect.getsourcelines(player_1)[0]),
            "player_2": str(inspect.getsourcelines(player_2)[0]),
            "sentences": sentences,
            "history": history[self.memory_size:],
        }
        with open(self.out_file_name, "r") as in_file:
            json_data = json.load(in_file)

        json_data.append(data)
        with open(self.out_file_name, "w") as out_file:
            json.dump(json_data, out_file)
