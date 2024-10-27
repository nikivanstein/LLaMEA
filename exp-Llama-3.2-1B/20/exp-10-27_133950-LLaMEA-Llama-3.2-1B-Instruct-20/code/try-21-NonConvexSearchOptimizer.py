import numpy as np
import random
from scipy.optimize import minimize

class NonConvexSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func, mutation_rate=0.01, exploration_rate=0.2):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            # Initialize population with random solutions
            population = [random.uniform(self.search_space) for _ in range(100)]

            # Perform evolutionary strategies
            for _ in range(100):
                # Select fittest individual
                fittest_index = np.argmax(self.func_evaluations)
                fittest_individual = population[fittest_index]

                # Select new individual using Non-Convex Search
                new_individual = fittest_individual
                while True:
                    # Evaluate fitness
                    fitness = self.evaluate_fitness(new_individual)

                    # Select new individual using mutation and exploration
                    if random.random() < exploration_rate:
                        mutation = random.uniform(-1, 1)
                        new_individual = new_individual + mutation
                    else:
                        new_individual = new_individual + 1

                    # Check if new individual is within bounds
                    if new_individual < -5.0 or new_individual > 5.0:
                        break

                    # Update new individual's position
                    new_individual = new_individual * (self.search_space[1] - self.search_space[0]) + self.search_space[0]

                # Add new individual to population
                population.append(new_individual)

            # Return best individual
            return population[np.argmax(self.func_evaluations)]

        except Exception as e:
            print(f"Error: {e}")
            return None

    def evaluate_fitness(self, individual):
        # Replace this with your own evaluation function
        return individual**2

# Example usage:
optimizer = NonConvexSearchOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)