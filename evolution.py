from copy import deepcopy
from player import Player
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pickle
import os
import random

################ Selection Strategies ################

class SelectionStrategy(ABC):
    
    @abstractmethod
    def select(self, population: List[Player], num_selection: int) -> List[Player]:
        pass

class KBestSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
        new_population = sorted(population, key=lambda x: x.fitness, reverse=True)[:num_selection]
        if clone:
            new_population = [player.clone() for player in new_population]
        return new_population
        

class RouletteWheelSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
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
            selected_player = selected_player if not clone else selected_player.clone()
            next_population.append(selected_player)
        return next_population

class SUSSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
        # calculate accumulated fitness
        sum_fitness = 0
        accumulated_fitness = []
        for player in population:
            sum_fitness += player.fitness
            accumulated_fitness.append(sum_fitness)
        accumulated_fitness = [acc/sum_fitness for acc in accumulated_fitness]
        # create second ruler
        second_ruler_start = np.random.uniform(0, 1/num_selection) # 0.04
        second_ruler = np.linspace(second_ruler_start, 1, num_selection)
        # find the player with the random number
        next_population = []
        i = 0
        for j in range(len(second_ruler)):
            if second_ruler[j] >= accumulated_fitness[i]:
                i += 1
            selected_player = population[i] if not clone else population[i].clone()
            next_population.append(selected_player)
        return next_population

class QTournamentSelectionStrategy(SelectionStrategy):
    def __init__(self, q) -> None:
        super().__init__()
        self.q = q
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
        # select q players
        next_population = []
        # print('fit: ', [player.fitness for player in population])
        for i in range(num_selection):
            # select q players
            selected_players = np.random.choice(population, self.q, replace=False)
            # find the best player
            best_player = selected_players[0]
            for i, player in enumerate(selected_players):
                if player.fitness > best_player.fitness:
                    best_player = player
            
            # add the best player to the next population
            best_player = best_player if not clone else best_player.clone()
            next_population.append(best_player)
        # print('next: ', [player.fitness for player in next_population])
        return next_population

class RandomUniformSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
        # select players
        next_population = []
        for i in range(num_selection):
            # select a random player
            selected_player = population[np.random.randint(0, len(population))]
            # add the selected player to the next population
            selected_player = selected_player if not clone else selected_player.clone()
            next_population.append(selected_player)
        return next_population

class AllSelectionStrategy(SelectionStrategy):
    
    def select(self, population: List[Player], num_selection: int, clone: bool = False) -> List[Player]:
        if clone:
            population = [player.clone() for player in population]
        return population

################ Crossover Strategies ################

class CrossoverStrategy(ABC):
    
    def __init__(self, crossover_p: float) -> None:
        self.crossover_p = crossover_p

    def has_crossover(self) -> bool:
        p = np.random.uniform(0, 1)
        return p < self.crossover_p

    @abstractmethod
    def crossover(self, parent1: Player, parent2: Player) -> Tuple[Player]:
        pass

class ArithmeticCrossoverStrategy(CrossoverStrategy):
    
    def __init__(self, crossover_p: float, alpha: float = 0.3) -> None:
        super().__init__(crossover_p)
        self.alpha = alpha

    def crossover(self, parent1: Player, parent2: Player) -> Tuple[Player]:
        if not self.has_crossover():
            return parent1, parent2
        # create new player
        new_player1, new_player2 = Player(parent1.game_mode), Player(parent2.game_mode)
        # update weights
        for i in range(len(parent1.nn.weights)):
            new_player1.nn.weights[i] = self.alpha * parent1.nn.weights[i] + (1 - self.alpha) * parent2.nn.weights[i]
            new_player2.nn.weights[i] = self.alpha * parent2.nn.weights[i] + (1 - self.alpha) * parent1.nn.weights[i]
        # update biases
        for i in range(len(parent1.nn.biases)):
            new_player1.nn.biases[i] = self.alpha * parent1.nn.biases[i] + (1 - self.alpha) * parent2.nn.biases[i]
            new_player2.nn.biases[i] = self.alpha * parent2.nn.biases[i] + (1 - self.alpha) * parent1.nn.biases[i]
        return new_player1, new_player2

class BLXAlphaCrossoverStrategy(CrossoverStrategy):
        
    def __init__(self, crossover_p: float, alpha = 0.15) -> None:
        super().__init__(crossover_p)
        self.alpha = alpha

    def crossover(self, parent1: Player, parent2: Player) -> Tuple[Player]:
        if not self.has_crossover():
            return parent1, parent2
        # create new player
        new_player1, new_player2 = Player(parent1.game_mode), Player(parent2.game_mode)
        # update weights
        for i in range(len(parent1.nn.weights)):
            min_w = np.minimum(parent1.nn.weights[i], parent2.nn.weights[i])
            max_w = np.maximum(parent1.nn.weights[i], parent2.nn.weights[i])
            w_range = max_w - min_w
            new_player1.nn.weights[i] = np.random.uniform(min_w - self.alpha * w_range, max_w + self.alpha * w_range)
            new_player2.nn.weights[i] = np.random.uniform(min_w - self.alpha * w_range, max_w + self.alpha * w_range)
        # update biases
        for i in range(len(parent1.nn.biases)):
            min_b = np.minimum(parent1.nn.biases[i], parent2.nn.biases[i])
            max_b = np.maximum(parent1.nn.biases[i], parent2.nn.biases[i])
            range_b = max_b - min_b
            new_player1.nn.biases[i] = np.random.uniform(min_b - self.alpha * range_b, max_b + self.alpha * range_b)
            new_player2.nn.biases[i] = np.random.uniform(min_b - self.alpha * range_b, max_b + self.alpha * range_b)
        return new_player1, new_player2

class TwoPointCrossoverStrategy(CrossoverStrategy):
    
    def crossover(self, parent1: Player, parent2: Player) -> Tuple[Player]:
        if not self.has_crossover():
            return parent1, parent2
        # create new player
        new_player1, new_player2 = Player(parent1.game_mode), Player(parent2.game_mode)
        for i in range(len(parent1.nn.weights)):
            weights_shape = parent1.nn.weights[i].shape
            new_player1.nn.weights[i] = deepcopy(parent1.nn.weights[i])
            new_player2.nn.weights[i] = deepcopy(parent2.nn.weights[i])
            # swap the weights
            weights1_flatten = new_player1.nn.weights[i].flatten()
            weights2_flatten = new_player2.nn.weights[i].flatten()
            # select two random points
            point1 = np.random.randint(0, weights1_flatten.shape[0])
            point2 = np.random.randint(0, weights1_flatten.shape[0])
            point1, point2 = min(point1, point2), max(point1, point2)
            # swap weights
            weights1_flatten[point1:point2], weights2_flatten[point1:point2] = weights2_flatten[point1:point2], weights1_flatten[point1:point2]
            # reshape weights
            new_player1.nn.weights[i] = weights1_flatten.reshape(weights_shape)
            new_player2.nn.weights[i] = weights2_flatten.reshape(weights_shape)
        for i in range(len(parent1.nn.biases)):
            biases_shape = parent1.nn.biases[i].shape
            new_player1.nn.biases[i] = deepcopy(parent1.nn.biases[i])
            new_player2.nn.biases[i] = deepcopy(parent2.nn.biases[i])
            # swap the biases
            biases1_flatten = new_player1.nn.biases[i].flatten()
            biases2_flatten = new_player2.nn.biases[i].flatten()
            # select two random points
            point1 = np.random.randint(0, biases1_flatten.shape[0])
            point2 = np.random.randint(0, biases1_flatten.shape[0])
            point1, point2 = min(point1, point2), max(point1, point2)
            # swap biases
            biases1_flatten[point1:point2], biases2_flatten[point1:point2] = biases2_flatten[point1:point2], biases1_flatten[point1:point2]
            # reshape biases
            new_player1.nn.biases[i] = biases1_flatten.reshape(biases_shape)
            new_player2.nn.biases[i] = biases2_flatten.reshape(biases_shape)
        return new_player1, new_player2

################ Mutation Strategies ################

class MutationStrategy(ABC):
    
    def __init__(self, mutation_p: float) -> None:
        self.mutation_p = mutation_p

    def has_mutation(self) -> bool:
        p = np.random.uniform(0, 1)
        return p < self.mutation_p

    @abstractmethod
    def mutation(self, player: Player) -> Player:
        pass

class GaussianMutationStrategy(MutationStrategy):
    
    def __init__(self, mutation_p: float, mu: float = 0, sigma: float = 1) -> None:
        super().__init__(mutation_p)
        self.mu = mu
        self.sigma = sigma

    def mutation(self, player: Player) -> Player:
        # update weights
        for i in range(len(player.nn.weights)):
            for y in range(player.nn.weights[i].shape[0]):
                for x in range(player.nn.weights[i].shape[1]):
                    if self.has_mutation():
                        player.nn.weights[i][y, x] += player.nn.weights[i][y, x] * 0.3 * np.random.normal(self.mu, self.sigma)

        # update biases
        for i in range(len(player.nn.biases)):
            for y in range(player.nn.biases[i].shape[0]):
                if self.has_mutation():
                    player.nn.biases[i][y, 0] += player.nn.biases[i][y, 0] * 0.3 * np.random.normal(self.mu, self.sigma)
        return player

class Evolution:
    def __init__(self, next_population_strategy: SelectionStrategy = QTournamentSelectionStrategy(20), \
                parent_selection_strategy: SelectionStrategy = SUSSelectionStrategy(), \
                crossover_strategy: CrossoverStrategy = TwoPointCrossoverStrategy(crossover_p=0.75),
                mutation_strategy: MutationStrategy = GaussianMutationStrategy(mutation_p=0.1, sigma = 1)) -> None:
        self.game_mode = "Neuroevolution"
        self.next_population_strategy = next_population_strategy
        self.parent_selection_strategy = parent_selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.data = {
            'max_fitness': [],
            'min_fitness': [],
            'avg_fitness': [],
        }

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # creating next population
        next_population = self.next_population_strategy.select(players, num_players, clone=True)  
        # extract data
        all_fitness = [player.fitness for player in next_population]
        min_fitness = min(all_fitness)
        max_fitness = max(all_fitness)
        avg_fitness = sum(all_fitness) / len(all_fitness)
        # add data to global variable
        self.data['min_fitness'].append(min_fitness)
        self.data['max_fitness'].append(max_fitness)
        self.data['avg_fitness'].append(avg_fitness)

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
            parents = self.parent_selection_strategy.select(prev_players, num_players, clone=True)
            random.shuffle(parents)
            # crossover
            children = []
            for i in range(0, num_players, 2):
                child1, child2 = self.crossover_strategy.crossover(parents[i], parents[i + 1])
                children.append(child1)
                children.append(child2)  
            # print("------------------------------------------------------------------")
            children = [self.mutation_strategy.mutation(child) for child in children]
            return children

    def save_data(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/evolution.pkl', 'wb') as f:
            pickle.dump(self.data, f)

    def print_data(self, players: list, name: str, sort=True):
        if sort:
            population = sorted(players, key=lambda x: x.fitness, reverse=True)
        else:
            population = players
        f_list = [p.fitness for p in population]
        print(name, f_list)