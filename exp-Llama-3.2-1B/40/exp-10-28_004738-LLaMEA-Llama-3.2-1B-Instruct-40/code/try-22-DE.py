import random
import numpy as np

class DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, population_size):
        # Initialize population with random individuals
        population = np.random.rand(population_size, self.dim)
        for _ in range(self.budget):
            # Evaluate fitness of each individual
            fitnesses = self.evaluate_fitness(population)
            # Select parents using tournament selection
            parents = np.array([self.select_parents(population, fitnesses, bounds) for _ in range(population_size // 2)])
            # Crossover (mate) and mutate
            offspring = np.array([self.crossover(parents, fitnesses) for _ in range(population_size // 2)])
            # Replace worst individuals with new offspring
            population[(population < offspring).any(axis=1)] = offspring
        return population

    def evaluate_fitness(self, population):
        fitnesses = []
        for individual in population:
            func_value = self.funcs[int(individual)](individual)
            fitnesses.append(func_value)
        return np.array(fitnesses)

    def select_parents(self, population, fitnesses, bounds):
        # Select parents using tournament selection
        parents = []
        for _ in range(len(population) // 2):
            tournament = np.array([self.funcs[int(random.choice(range(len(population))))](random.uniform(bounds[0], bounds[1])) for _ in range(3)])
            tournament_fitnesses = self.evaluate_fitness(tournament)
            winner = tournament[np.argmin(tournament_fitnesses)]
            parents.append(winner)
        return np.array(parents)

    def crossover(self, parents, fitnesses):
        # Crossover (mate) using uniform crossover
        offspring = np.zeros_like(parents)
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                if random.random() < 0.5:
                    offspring[i] = parents[i]
                    offspring[j] = parents[j]
                else:
                    idx = random.randint(0, len(parents[i]) - 1)
                    offspring[i][idx] = parents[j][idx]
        return offspring

# Description: Evolutionary Optimization using Differential Evolution
# Code: 