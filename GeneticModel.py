import random
from typing import Callable, List, Any
import numpy as np
class Gene:
    def __init__(self, values: list, objective_fun: Callable[list, float]):
        self.values = values
        self.objective_val = objective_fun(values)

    def copy(self):
        return Gene(self.values.copy(), lambda values: self.objective_val)


class GeneticAlgorithmModel:
    def __init__(self, gene_size: int, population_size: int = 50):
        self.__crossover_fun: Callable[[Gene, Gene], (Gene, Gene)] = None
        self.__mutation_fun: Callable[[Gene], Gene] = None
        self.__crossover_num = 0
        self.__mutation_num = 0
        self.__random_population_fun: Callable[[int], list[Gene]] = None
        self.__gene_size: int = gene_size
        self.__population_size: int = population_size
        self.population: list[Gene] = []
        self.objectives: list[float] = []
        self.__objectives_sum: float = 0

    def compile(self, crossover_fun, mutation_fun, random_population_fun, crossover_coeff, mutation_coeff):
        self.__crossover_fun = crossover_fun
        self.__mutation_fun = mutation_fun
        self.__random_population_fun = random_population_fun
        self.__crossover_num = self.__population_size * crossover_coeff
        self.__mutation_num = self.__population_size * mutation_coeff

    def __choose_weighted(self, k):
        objectives_sum = sum(self.objectives)
        return np.random.choice(
            self.population, size=k, replace=False,
            p=[u / objectives_sum for u in self.objectives])

    def __choose_best(self, k):
        l1 = self.population.copy()
        l1.sort(key=lambda e: e.objective_val)
        return l1[-k:]

    def extend(self, genes):
        self.population.extend(genes)
        self.objectives.extend([t.objective_val for t in genes])

    def remove(self, gene):
        i1 = self.population.index(gene)
        self.population.pop(i1)
        self.objectives.pop(i1)

    def __print_on_epoch(self, epoch, metrics):
        print('Epoch #' + str(epoch) + ' -->', end='')
        if 'best_objective' in metrics:
            t = np.argmax(self.objectives)
            print(' best_objective_val=', self.objectives[t], end='')
            print(' ,best_objective=', self.population[t].values, end='')
        print('\n')

    def fit(self, epochs, metrics=[]):
        epoch = 1
        self.population = self.__random_population_fun(self.__population_size)
        self.objectives = [t.objective_val for t in self.population]
        self.__print_on_epoch(0, metrics)
        best_objectives = []
        best_objectives.append(self.__choose_best(1)[0].objective_val)
        while epoch <= epochs:

            crossover_index = 0
            while crossover_index < self.__crossover_num:
                parent1, parent2 = self.__choose_weighted(2)
                child1, child2 = self.__crossover_fun(parent1.copy(), parent2.copy())
                self.extend([child1, child2])
                if 'crossovers' in metrics:
                    print(
                        f'Crossovering:\n  {parent1.values} \n  {parent2.values} \n  childs: \n  {child1.values} \n  {child2.values}')
                crossover_index += 2

            mutation_index = 0
            while mutation_index < self.__mutation_num:
                to_mutate = random.choice(self.population)
                mutated = self.__mutation_fun(to_mutate.copy())
                self.extend([mutated])
                if 'mutates' in metrics:
                    print(f'Mutating:\n  {to_mutate.values} \n  mutated: \n  {mutated.values}')
                mutation_index += 1

            self.population = self.__choose_best(self.__population_size)
            self.objectives = [t.objective_val for t in self.population]

            self.__print_on_epoch(epoch, metrics)
            best_objectives.append(self.__choose_best(1)[0].objective_val)
            epoch += 1

        return best_objectives
