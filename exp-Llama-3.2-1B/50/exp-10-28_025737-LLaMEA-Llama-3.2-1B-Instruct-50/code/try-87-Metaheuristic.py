import random
import numpy as np
from scipy.optimize import minimize

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize the population with random individuals
        population = [self.evaluate_fitness(np.random.uniform(self.search_space)) for _ in range(100)]

        # Define the mutation operator
        def mutate(individual):
            if random.random() < 0.45:
                index = random.randint(0, self.dim - 1)
                individual[index] += np.random.uniform(-5.0, 5.0)
            return individual

        # Evolve the population over iterations
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitnesses = [self.evaluate_fitness(individual) for individual in population]

            # Select the fittest individuals
            fittest = [individual for individual, fitness in zip(population, fitnesses) if fitness == fitnesses[0]]

            # Create a new population by mutation
            new_population = [mutate(individual) for individual in fittest]

            # Replace the old population with the new one
            population = new_population

            # If the budget is exhausted, stop evolving
            if len(fittest) == 0:
                break

        # Select the best individual
        best_individual = max(set(fittest), key=fittest.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_individual]

        return best_individual

    def evaluate_fitness(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func