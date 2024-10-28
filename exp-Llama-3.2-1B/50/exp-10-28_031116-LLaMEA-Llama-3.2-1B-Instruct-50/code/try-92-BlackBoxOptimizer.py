import numpy as np
from scipy.optimize import minimize
from random import randint

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be negative")
        if self.dim == 0:
            return func(np.random.rand(1))  # Simple random search
        if self.dim == 1:
            return func(np.random.rand(1))  # Simple random search
        if self.dim == 2:
            return func(np.random.rand(2))  # Simple random search
        if self.dim == 3:
            return func(np.random.rand(3))  # Simple random search
        if self.dim == 4:
            return func(np.random.rand(4))  # Simple random search
        if self.dim == 5:
            return func(np.random.rand(5))  # Simple random search

        # Novel Metaheuristic Algorithm: Genetic Algorithm with Adaptive Probability of Mutation
        def genetic_algorithm(func, population_size, mutation_rate):
            while True:
                # Initialize population with random solutions
                population = [func(np.random.rand()) for _ in range(population_size)]

                # Evaluate population and select fittest solutions
                fitnesses = [func(x) for x in population]
                fittest_indices = np.argsort(fitnesses)[:self.budget]
                fittest_solutions = [population[i] for i in fittest_indices]

                # Calculate mutation probability based on budget and population size
                mutation_prob = min(0.5, self.budget / population_size)

                # Select new solutions by mutation
                new_solutions = []
                for _ in range(population_size):
                    if np.random.rand() < mutation_prob:
                        # Randomly select a solution from the current population
                        idx = np.random.randint(0, self.budget)
                        solution = population[idx]
                        # Mutate the solution by adding a small random perturbation
                        perturbation = np.random.normal(0, 0.1, self.dim)
                        new_solution = solution + perturbation
                        # Add new solution to the new solutions list
                        new_solutions.append(new_solution)

                # Replace the current population with the new solutions
                population = new_solutions

                # Select fittest solutions again
                fitnesses = [func(x) for x in population]
                fittest_indices = np.argsort(fitnesses)[:self.budget]
                fittest_solutions = [population[i] for i in fittest_indices]

                # Check if the new population is better than the current best
                if np.max(fitnesses) > np.max([func(x) for x in fittest_solutions]):
                    # Update the best solution
                    best_solution = fittest_solutions[0]
                    best_fitness = np.max(fitnesses)
                    best_func = func

                    # Refine the strategy based on the number of iterations
                    if self.dim == 1:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand()) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 2:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(2)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 3:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(3)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 4:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(4)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 5:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(5)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1

                    # Update the best solution
                    best_solution = fittest_solutions[0]
                    best_fitness = np.max(fitnesses)
                    best_func = func

                    # Refine the strategy based on the number of iterations
                    if self.dim == 1:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(1)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 2:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(2)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 3:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(3)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 4:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(4)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1
                    elif self.dim == 5:
                        # Simple strategy: increase population size and mutation rate
                        self.population = [func(np.random.rand(5)) for _ in range(10 * population_size)]
                        mutation_rate = 0.1

                # Update the best solution
                best_solution = fittest_solutions[0]
                best_fitness = np.max(fitnesses)
                best_func = func

                # Refine the strategy based on the number of iterations
                if self.dim == 1:
                    # Simple strategy: increase population size and mutation rate
                    self.population = [func(np.random.rand(1)) for _ in range(10 * population_size)]
                    mutation_rate = 0.1
                elif self.dim == 2:
                    # Simple strategy: increase population size and mutation rate
                    self.population = [func(np.random.rand(2)) for _ in range(10 * population_size)]
                    mutation_rate = 0.1
                elif self.dim == 3:
                    # Simple strategy: increase population size and mutation rate
                    self.population = [func(np.random.rand(3)) for _ in range(10 * population_size)]
                    mutation_rate = 0.1
                elif self.dim == 4:
                    # Simple strategy: increase population size and mutation rate
                    self.population = [func(np.random.rand(4)) for _ in range(10 * population_size)]
                    mutation_rate = 0.1
                elif self.dim == 5:
                    # Simple strategy: increase population size and mutation rate
                    self.population = [func(np.random.rand(5)) for _ in range(10 * population_size)]
                    mutation_rate = 0.1

        return best_func

# Example usage:
optimizer = BlackBoxOptimizer(budget=100, dim=5)
best_func = optimizer(genetic_algorithm)

# Print the best solution and its score
print(f"Best solution: {best_func(np.random.rand(5))}")
print(f"Score: {np.max([best_func(x) for x in range(-10, 10)])}")