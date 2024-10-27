# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        if budget < 0:
            raise ValueError("Invalid budget")
        if budget == 0:
            return None

        # Initialize the population with random individuals
        population = [self.evaluate_fitness(x) for x in self.generate_individuals(self.budget)]

        # Initialize the best solution
        best_individual = population[0]
        best_fitness = self.evaluate_fitness(best_individual)

        while len(population) > 1:
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.evaluate_fitness, reverse=True)[:self.budget]

            # Create a new generation by refining the fittest individuals
            new_individuals = []
            for _ in range(self.budget):
                individual = fittest_individuals[np.random.randint(0, len(fittest_individuals))]
                updated_individual = self.f(individual, self.func, budget)
                if updated_individual is not None:
                    new_individuals.append(updated_individual)

            # Replace the old population with the new one
            population = new_individuals

            # Update the best solution
            best_individual = population[0]
            best_fitness = self.evaluate_fitness(best_individual)

            # Refine the search space
            self.search_space = np.vstack((self.search_space, [x for x in self.search_space if x not in best_individual]))
            self.search_space = np.delete(self.search_space, 0, axis=0)

        return best_individual

    def generate_individuals(self, budget):
        return np.random.uniform(self.search_space, size=(budget, self.dim))

    def f(self, individual, func, budget):
        return func(individual)

    def evaluate_fitness(self, individual):
        return self.func(individual)