import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        # Initialize population with random solutions
        population = [self.evaluate_fitness(individual) for individual in self.generate_population(self.budget)]

        # Simulated Annealing
        temperature = 1.0
        while temperature > 0.5:
            # Select the best solution in the population
            best_solution = max(population)
            # Select a random solution with a lower fitness
            new_individual = self.generate_random_solution()
            # Calculate the fitness of the new individual
            new_fitness = self.evaluate_fitness(new_individual)
            # Calculate the probability of accepting the new individual
            prob_accept = np.exp((new_fitness - best_solution) / temperature)
            # Accept the new individual with a probability less than or equal to 0.95
            if random.random() <= prob_accept:
                population.remove(new_individual)
                population.append(new_individual)
            # Decrease the temperature for the next iteration
            temperature *= 0.95

        return population

    def generate_population(self, budget):
        population = []
        for _ in range(budget):
            individual = self.generate_random_individual(self.dim)
            population.append(individual)
        return population

    def generate_random_individual(self, dim):
        return np.random.uniform(-5.0, 5.0, size=(dim,))

    def evaluate_fitness(self, individual):
        return self.func(individual)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 