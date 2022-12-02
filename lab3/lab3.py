from copy import deepcopy
import random
import itertools
from typing import Iterable, List, Optional
from two_player_games.game import Game
from two_player_games.move import Move
from two_player_games.player import Player
from two_player_games.state import State


class Pick(Game):
    """Class that represents the Pick game"""
    FIRST_PLAYER_DEFAULT_CHAR = '1'
    SECOND_PLAYER_DEFAULT_CHAR = '2'

    def __init__(self, first_player: Player = None, second_player: Player = None, n: int = 4):
        """
        Initializes game.

        Parameters:
            first_player: the player that will go first (if None is passed, a player will be created)
            second_player: the player that will go second (if None is passed, a player will be created)
            n: the subset size of picked numbers that should sum up to aim value
        """

        self.first_player = first_player or Player(self.FIRST_PLAYER_DEFAULT_CHAR)
        self.second_player = second_player or Player(self.SECOND_PLAYER_DEFAULT_CHAR)
        state = PickState(self.first_player, self.second_player, n)
        super().__init__(state)
        self.all_combinations = self.all_unique_moves()

    def all_unique_moves(self):
        candidates = range(1, self.state.max_number + 1)

        def backtrack(i, target):
            combinations = []
            for index in range(i, len(candidates)):
                num = candidates[index]
                if index > i and num == candidates[index - 1]:
                    continue
                if num < target:
                    combinations.extend([num] + comb for comb in backtrack(index + 1, target - num))
                else:
                    if num == target:
                        combinations.append([num])
                    break
            return combinations

        all_combination = backtrack(0, self.state.aim_value)
        all_combination = [item for item in all_combination if len(item) == self.state.n]

        return all_combination


class PickMove(Move):
    """
    Class that represents a move in the PickMove game

    Variables:
        number: selected number (from 1 to n^2)
    """

    def __init__(self, number: int):
        self.number = number

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, PickMove):
            return False
        return self.number == o.number

    def __str__(self) -> str:
        return f"{self.number}"


class PickState(State):
    """Class that represents a state in the PickState game"""

    def __init__(self,
                 current_player: Player, other_player: Player, n,
                 current_player_numbers: List[int] = None,
                 other_player_numbers: List[int] = None):
        """Creates the state. Do not call directly."""

        if current_player_numbers is None:
            current_player_numbers = []
        if other_player_numbers is None:
            other_player_numbers = []

        self.current_player_numbers = current_player_numbers
        self.other_player_numbers = other_player_numbers
        self.selected_numbers = set(self.current_player_numbers).union(self.other_player_numbers)
        self.n = n
        self.max_number = n ** 2
        self.aim_value = int((n ** 2 * (n ** 2 + 1)) / (2 * n))
        super().__init__(current_player, other_player)

    def get_moves(self) -> Iterable[PickMove]:
        return [PickMove(number) for number in range(1, self.max_number + 1) if number not in self.selected_numbers]

    def make_move(self, move: PickMove) -> 'PickState':
        if move.number > self.max_number or move.number in self.selected_numbers:
            raise ValueError("Invalid move")
        else:
            next_player = self._other_player
            next_player_numbers = self.other_player_numbers

            other_player = self._current_player
            other_player_numbers = self.current_player_numbers + [move.number]

        return PickState(
            next_player, other_player, self.n, next_player_numbers, other_player_numbers
        )

    def is_finished(self) -> bool:
        return self._check_if_sums_to_aim_value(self.current_player_numbers) or \
               self._check_if_sums_to_aim_value(self.other_player_numbers) or \
               len(self.selected_numbers) == self.max_number

    def get_winner(self) -> Optional[Player]:
        if not self.is_finished():
            return None
        if self._check_if_sums_to_aim_value(self.current_player_numbers):
            return self._current_player
        elif self._check_if_sums_to_aim_value(self.other_player_numbers):
            return self._other_player
        else:
            return None

    def __str__(self) -> str:
        return f"n: {self.n}, aim_value: {self.aim_value}" \
               f"\nCurrent player: {self._current_player.char}, Numbers: " \
               f"{'[]' if not self.current_player_numbers else sorted(self.current_player_numbers)}," \
               f"\nOther player: {self._other_player.char}, Numbers: " \
               f"{'[]' if not self.other_player_numbers else sorted(self.other_player_numbers)}"

    # below are helper methods for the public interface

    def _check_if_sums_to_aim_value(self, numbers: List[int]) -> bool:
        return self.aim_value in [sum(i) for i in itertools.combinations(numbers, self.n)]


def heuristic(game, maximizing_player):
    if game.is_finished():
        if game.get_winner():
            if maximizing_player:
                return -100000000000000000
            else:
                return 100000000000000000
        else:
            return 0

    current_player_moves = game.state.current_player_numbers
    possible_moves = [item.number for item in game.state.get_moves()]

    new_combinations = []
    score = 0

    for combination in game.all_combinations:
        good = True
        for item in combination:
            if item not in possible_moves and item not in current_player_moves:
                good = False
                break

        if good:
            new_combinations.append(combination)

    for combination in new_combinations:
        count = 0
        for item in combination:
            if item in current_player_moves:
                count += 1
        score += pow(10, count)

    if maximizing_player:
        return -score
    return score


def minimax(game: Pick, depth, alpha, beta, maximizing_player):
    if depth == 0 or game.state.is_finished():
        return [heuristic(game, maximizing_player), 0]

    if maximizing_player:
        max_val = float('-inf')
        moves = {}

        for possible_move in game.get_moves():
            game_copy = deepcopy(game)
            game_copy.make_move(possible_move)

            [val, pos] = minimax(game_copy, depth - 1, alpha, beta, False)

            if pos == 0:
                return [val, possible_move]

            if val > max_val:
                max_val = val
                moves = {f"{max_val}": [possible_move]}
            elif val == max_val:
                moves[f"{max_val}"].append(possible_move)

            alpha = max(alpha, max_val)

            if beta <= alpha:
                break
        return [max_val, random.choice(moves[f"{max_val}"])]
    else:
        min_val = float('inf')
        moves = {}

        for possible_move in game.get_moves():
            game_copy = deepcopy(game)
            game_copy.make_move(possible_move)

            [val, pos] = minimax(game_copy, depth - 1, alpha, beta, True)

            if pos == 0:
                return [val, possible_move]

            if val < min_val:
                min_val = val
                moves = {f"{min_val}": [possible_move]}
            elif val == min_val:
                moves[f"{min_val}"].append(possible_move)

            beta = min(beta, min_val)

            if beta <= alpha:
                break
        return [min_val, random.choice(moves[f"{min_val}"])]


def two_computers_game(size, depth_arr, p=False):
    game = Pick(n=size)
    player = 0

    while not game.state.is_finished():
        [score, move] = minimax(game, depth_arr[player], float('-inf'), float('inf'), True)

        if p:
            print(f"player: {player+1}, score value: {score}, move: {move.number}, PLAYER_MOVES: {game.state.current_player_numbers}")

        game.make_move(move)

        player = 0 if player == 1 else 1

    if p:
        if game.state.get_winner():
            print("Winner: ", game.state.get_winner().char)
        else:
            print("Remis")

    return game.state.get_winner()


def get_human_move(game):
    possible_moves = [move.number for move in game.state.get_moves()]

    print(f"Game has such possible moves: {possible_moves}")

    move = int(input("Enter move: "))

    while move not in possible_moves:
        print("Error :0")
        print("Please select correct move: ")
        move = int(input("Enter move: "))

    return PickMove(move)


def get_ai_move(game, depth):
    [_, move] = minimax(game, depth, float('-inf'), float('inf'), True)
    return move


def play(size, depth):
    game = Pick(n=size)

    while not game.state.is_finished():
        human_move = get_human_move(game)
        game.make_move(human_move)

        if game.state.is_finished():
            break

        ai_move = get_ai_move(game, depth)
        game.make_move(ai_move)
        print(f"Ai move: {ai_move.number}")

    if game.get_winner():
        if game.get_winner().char == '1':
            print("You won !!!!")
        else:
            print("AI won :)")
    else:
        print("Draw")


def main():
    size = int(input("Enter game size (don't set to big 3-5): "))
    depth = int(input("Enter ai level (don't set to big 1-5): "))

    play(size, depth)


if __name__ == "__main__":
    main()
