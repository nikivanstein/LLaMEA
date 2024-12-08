import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100, max_iter=1000, mutation_prob=0.05):
        """
        Optimizes the black box function `func` using the Novel Metaheuristic Algorithm for Black Box Optimization.

        Args:
            func (function): The black box function to optimize.
            budget (int, optional): The number of function evaluations allowed. Defaults to 100.
            max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
            mutation_prob (float, optional): The probability of mutation. Defaults to 0.05.

        Returns:
            tuple: The optimized individual and its fitness score.
        """
        # Initialize the population with random individuals
        population = self.initialize_population(budget, max_iter, mutation_prob)

        # Initialize the best individual and its fitness score
        best_individual = None
        best_fitness = -np.inf

        # Iterate until the maximum number of iterations is reached
        for _ in range(max_iter):
            # Select the best individual
            current_individual = population[np.argmax([individual.fitness for individual in population])]

            # Evaluate the fitness of the current individual
            current_fitness = current_individual.fitness

            # If the current individual is better than the best individual, update the best individual
            if current_fitness > best_fitness:
                best_individual = current_individual
                best_fitness = current_fitness

            # If the budget is exhausted, stop the algorithm
            if _ == budget - 1:
                break

            # Generate a new individual by mutating the best individual
            mutation_prob = mutation_prob / 2
            new_individual = current_individual.copy()
            for _ in range(self.dim):
                if random.random() < mutation_prob:
                    new_individual[-1] += np.random.uniform(-1, 1)
                elif random.random() < mutation_prob:
                    new_individual[-1] -= np.random.uniform(-1, 1)

            # Add the new individual to the population
            population.append(new_individual)

        # Return the optimized individual and its fitness score
        return best_individual, best_fitness

    def initialize_population(self, budget, max_iter, mutation_prob):
        """
        Initializes the population with random individuals.

        Args:
            budget (int): The number of function evaluations allowed.
            max_iter (int): The maximum number of iterations.
            mutation_prob (float): The probability of mutation.

        Returns:
            list: The initialized population.
        """
        population = []
        for _ in range(budget):
            individual = random.choice([self.search_space])
            for _ in range(self.dim):
                individual = np.append(individual, random.uniform(-5.0, 5.0))
            population.append(individual)
        return population

# Example usage:
optimizer = BBOBOptimizer(100, 2)
best_individual, best_fitness = optimizer(func, mutation_prob=0.05)
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")