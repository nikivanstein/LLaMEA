import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [np.random.uniform(self.search_space[0], self.search_space[1], self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def select_parents(self):
        parents = []
        for _ in range(self.population_size):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            while parent2 == parent1:
                parent2 = random.choice(self.population)
            parents.append((parent1, parent2))
        return parents

    def crossover(self, parents):
        children = []
        for parent1, parent2 in parents:
            if random.random() < self.crossover_rate:
                child1 = parent1[:self.dim//2] + parent2[self.dim//2:]
                child2 = parent2[:self.dim//2] + parent1[self.dim//2:]
            else:
                child1 = parent1
                child2 = parent2
            children.append(child1)
            children.append(child2)
        return children

    def mutate(self, children):
        mutated_children = []
        for child in children:
            if random.random() < self.mutation_rate:
                idx = random.randint(0, self.dim-1)
                child[idx] = random.uniform(self.search_space[0], self.search_space[1])
            mutated_children.append(child)
        return mutated_children

    def fitness(self, func, child):
        return np.mean(func(child))

    def selection(self, parents, children):
        fitnesses = [self.fitness(func, child) for child, func in zip(children, parents)]
        selected_parents = np.array([parents[i] for i, _ in sorted(zip(fitnesses, range(len(fitnesses)))) if fitnesses[i] > np.mean(fitnesses[:i+1])])
        selected_children = [children[i] for i, _ in sorted(zip(fitnesses, range(len(fitnesses)))) if fitnesses[i] > np.mean(fitnesses[:i+1])]
        return selected_parents, selected_children

    def evolve(self, selected_parents, selected_children):
        population = self.population
        for _ in range(self.budget):
            parents, children = self.selection(selected_parents, selected_children)
            children = self.crossover(parents)
            children = self.mutate(children)
            population = self.population + children
            self.func_values = np.zeros(self.dim)
            for func in population:
                self.func_values = np.mean(func(self.func_values))
            selected_parents, selected_children = self.selection(parents, children)
        return selected_parents

# Description: AdaptiveBlackBoxOptimizer
# Code: 