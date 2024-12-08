import random
import math

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        for _ in range(self.budget):
            # Adaptive sampling: select the next individual based on the fitness and the dimension
            # Use a simple strategy: select the individual with the highest fitness
            next_individual = self.select_next_individual()

            # Evaluate the function at the next individual
            fitness = func(next_individual)

            # Update the fitness and the population
            self.fitnesses[self.population_size - 1] += fitness
            self.population[self.population_size - 1] = next_individual

            # Ensure the fitness stays within the bounds
            self.fitnesses[self.population_size - 1] = min(max(self.fitnesses[self.population_size - 1], -5.0), 5.0)

        # Return the best individual
        return self.population[0]

    def select_next_individual(self):
        # Select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Use adaptive sampling to select the next individual based on the fitness and the dimension
        # Use a simple strategy: select the individual with the highest fitness
        # Use adaptive sampling to select the next individual based on the fitness and the dimension
        return max(self.population, key=lambda x: self.fitnesses[x])

    def mutate(self, individual):
        # Mutate the individual by a small random amount
        # Use a simple strategy: mutate the individual with a small random amount
        # Use a simple strategy: mutate the individual with a small random amount
        return random.uniform(-0.1, 0.1) + individual

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.

# Exception Occurred: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
#     algorithm(problem)
#   File "<string>", line 15, in __call__
#     File "<string>", line 33, in select_next_individual
#     File "<string>", line 33, in <lambda>
#     TypeError: list indices must be integers or slices, not float