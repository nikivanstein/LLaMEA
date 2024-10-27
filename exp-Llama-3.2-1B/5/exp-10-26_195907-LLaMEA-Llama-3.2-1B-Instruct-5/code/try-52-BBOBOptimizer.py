import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def optimize(self, func, iterations=100, mutation_rate=0.01, mutation_bound=2.0):
        """Optimize the black box function using the Novel Metaheuristic Algorithm"""
        # Initialize the population with random individuals
        population = [self.evaluate_fitness(func, individual) for _ in range(100)]

        # Evolve the population for the specified number of iterations
        for _ in range(iterations):
            # Select the fittest individuals to reproduce
            parents = self.select_parents(population)

            # Generate offspring by mutating the parents
            offspring = self.mutate(parents)

            # Replace the least fit individuals with the offspring
            population = self.replace_least_fit(population, offspring)

        # Evaluate the final population to estimate the fitness of the best individual
        best_individual = self.evaluate_fitness(func, population[0])
        return best_individual

    def select_parents(self, population):
        """Select the fittest individuals to reproduce"""
        # Calculate the fitness of each individual and sort it
        fitnesses = np.array([self.evaluate_fitness(func, individual) for individual in population])
        sorted_indices = np.argsort(fitnesses)

        # Select the top 50% of the individuals
        parents = population[sorted_indices[:len(population)//2]]

        return parents

    def mutate(self, parents):
        """Mutate the parents by changing a random individual"""
        mutated_parents = []
        for parent in parents:
            for _ in range(self.mutation_rate):
                x = parent[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    new_x = x + np.random.uniform(-self.budget/2, self.budget/2)
                    mutated_parents.append(new_x)
            mutated_parents.append(parent)
        return mutated_parents

    def replace_least_fit(self, population, offspring):
        """Replace the least fit individuals with the offspring"""
        # Sort the population by fitness
        sorted_indices = np.argsort(population)

        # Replace the least fit individuals with the offspring
        population[sorted_indices] = offspring

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Optimizes black box functions using a population-based approach with mutation and selection