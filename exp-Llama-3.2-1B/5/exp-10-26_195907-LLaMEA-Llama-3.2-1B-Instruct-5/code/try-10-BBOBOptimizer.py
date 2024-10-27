import random
import numpy as np
import operator

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        # Initialize population with random individuals
        population = self.generate_population(self.budget)

        while True:
            # Evaluate fitness of each individual
            fitnesses = [func(individual) for individual in population]

            # Select parents using tournament selection
            parents = self.select_parents(population, fitnesses, self.budget)

            # Create offspring using crossover and mutation
            offspring = self.crossover_and_mutation(parents, self.budget)

            # Replace old population with new offspring
            population = self.replace_population(population, offspring, self.budget)

            # Check for convergence
            if self.converged(population, fitnesses):
                return population

    def generate_population(self, budget):
        population = []
        for _ in range(budget):
            individual = np.random.uniform(self.search_space.shape[0], self.search_space.shape[1])
            population.append(individual)
        return population

    def select_parents(self, population, fitnesses, budget):
        # Select parents using tournament selection
        parents = []
        for _ in range(budget):
            tournament_size = random.randint(1, self.search_space.shape[0])
            winner_index = np.random.choice(self.search_space.shape[0], p=fitnesses)
            tournament = self.search_space[winner_index]
            for _ in range(tournament_size - 1):
                winner_index = np.random.choice(self.search_space.shape[0], p=fitnesses)
                tournament = np.vstack((tournament, self.search_space[winner_index]))
            parents.append(tournament)
        return parents

    def crossover_and_mutation(self, parents, budget):
        # Create offspring using crossover and mutation
        offspring = []
        for i in range(len(parents)):
            parent1, parent2 = parents[i], parents[(i + 1) % len(parents)]
            child = np.copy(parent1)
            for j in range(self.dim):
                if random.random() < 0.5:
                    child[j] = parent2[j]
            offspring.append(child)
        return offspring

    def replace_population(self, population, offspring, budget):
        # Replace old population with new offspring
        population = np.delete(population, 0, axis=0)
        population = np.vstack((population, offspring))
        return population

    def converged(self, population, fitnesses):
        # Check for convergence
        return all(np.linalg.norm(fitnesses - individual) < self.budget / 2 for individual in population)