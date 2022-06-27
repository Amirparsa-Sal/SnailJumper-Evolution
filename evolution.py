from dataclasses import replace

from player import Player
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class SelectionStrategy(ABC):
    
    @abstractmethod
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        pass

class KBestSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:num_selection]

class RouletteWheelSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        # calculate accumulated fitness
        sum_fitness = 0
        accumulated_fitness = []
        for player in population:
            sum_fitness += player.fitness
            accumulated_fitness.append(sum_fitness)
        # select players
        next_population = []
        for i in range(num_selection):
            # select a random number between 0 and sum_fitness
            random_number = sum_fitness * np.random.random()
            # find the player with the random number
            for j in range(len(accumulated_fitness)):
                if accumulated_fitness[j] >= random_number:
                    selected_player = population[j]
                    break
            # add the selected player to the next population
            next_population.append(selected_player)
        return next_population

class SUSSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        # calculate accumulated fitness
        sum_fitness = 0
        accumulated_fitness = []
        for player in population:
            sum_fitness += player.fitness
            accumulated_fitness.append(sum_fitness)
        # create second ruler
        second_ruler_start = np.random.random()
        second_ruler = [second_ruler_start + i / num_selection for i in range(num_selection)]
        # find the player with the random number
        next_population = []
        for i in range(len(accumulated_fitness)):
            if accumulated_fitness[i] >= second_ruler[i]:
                selected_player = population[i]
                break
        # add the selected player to the next population
        next_population.append(selected_player)
        return next_population

class QTournamentSelectionStrategy(SelectionStrategy):
    def __init__(self, q) -> None:
        super().__init__()
        self.q = q
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        # select q players
        next_population = []
        for i in range(num_selection):
            # select q players
            selected_players = np.random.choice(population, self.q, replace=True)
            # find the best player
            best_player = selected_players[0]
            for i, player in enumerate(selected_players):
                if player.fitness > best_player.fitness:
                    best_player = player
            
            # add the best player to the next population
            next_population.append(best_player)
        return next_population

class RandomUniformSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        # select players
        next_population = []
        for i in range(num_selection):
            # select a random player
            selected_player = population[np.random.randint(0, len(population))]
            # add the selected player to the next population
            next_population.append(selected_player)
        return next_population

class AllSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        return population

class Evolution:
    def __init__(self, next_population_strategy: SelectionStrategy = KBestSelectionStrategy(), \
                parent_selection_strategy: SelectionStrategy = AllSelectionStrategy()):
        self.game_mode = "Neuroevolution"
        self.next_population_strategy = next_population_strategy
        self.parent_selection_strategy = parent_selection_strategy

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        next_population = self.next_population_strategy.select(players, num_players)

        # TODO (Additional: Learning curve)
        return next_population

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # parent_selection
            parents = self.parent_selection_strategy.select(prev_players, num_players)
            # crossover
            # mutation
            new_players = prev_players  # DELETE THIS AFTER YOUR IMPLEMENTATION
            return new_players
