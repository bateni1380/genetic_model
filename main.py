import random
from typing import Callable, List, Any
import numpy as np
from GeneticModel import GeneticAlgorithmModel, Gene

with open('input.txt') as f:
    lines = f.readlines()
vertices_size = int(lines[0])
edges = np.zeros((vertices_size + 1, vertices_size + 1))
for line in lines:
    t = line.find(' ')
    if t > 0:
        edges[int(line[0:t])][int(line[t + 1:-1])] = True

def objective_fun(values):
    # Gets a gene and calculates its objective and returns nothing
    # Answer will be saved in gene.objective_val
    misses = 0.0
    for i in range(1, len(values)):
        misses += edges[values[i]][values[i - 1]]
    return 1 / (misses+1)

def random_perm_genes(n):
    # Gets an integer and returns a list of n genes
    population = []
    for i in range(n):
        solution = list(range(1, vertices_size + 1))
        random.shuffle(solution)
        gene = Gene(solution, objective_fun)
        population.append(gene)
    return population

def crossover(parent1, parent2):
    # Gets two genes and returns two genes
    if len(parent1.values) != len(parent2.values):
        raise Exception('Gene sizes are not equal!!')
    child1_values, child2_values = [], []
    for u in parent1.values:
        child1_values.append(parent2.values[u - 1])
    for u in parent2.values:
        child2_values.append(parent1.values[u - 1])

    return Gene(child1_values, objective_fun), Gene(child2_values, objective_fun)

def mutation(gene):
    # Gets a gene and returns a gene
    gene_size = len(gene.values)
    i1, i2 = np.random.choice(range(gene_size), 2, replace=False)
    gene_values = gene.values.copy()
    gene_values[i1], gene_values[i2] = gene_values[i2], gene_values[i1]
    return Gene(gene_values, objective_fun)


model = GeneticAlgorithmModel(gene_size=vertices_size, population_size=8)
model.compile(crossover, mutation, random_perm_genes
              , crossover_coeff=0.8, mutation_coeff=0.3)
r = model.fit(epochs=300, metrics=['best_objective']) # ,'mutates','crossovers', ...

import matplotlib.pyplot as plt
plt.plot(range(len(r)), r)
plt.xlabel('epochs')
plt.ylabel('objective value')
plt.show()