import random
import numpy as np

class EAPI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []
        self.population_size = 100

    def __call__(self, func):
        for _ in range(self.budget):
            # Select a random individual from the population
            individual = random.choices(self.population, k=1)[0]
            # Evaluate the function at the individual
            score = func(individual)
            # Add the score to the fitness scores
            self.fitness_scores.append(score)
            # Store the individual and its fitness score
            self.population.append(individual)
        # Select the fittest individuals
        self.population = self.population[:self.population_size]
        # Select the fittest individuals based on their fitness scores
        self.population = self.population[np.argsort(self.fitness_scores)]
        return self.population

    def mutate(self, individual):
        # Randomly select a dimension to mutate
        dim_to_mutate = random.randint(0, self.dim - 1)
        # Randomly select a value for the mutated dimension
        value = random.uniform(-5.0, 5.0)
        # Update the individual
        individual[dim_to_mutate] = value
        return individual

    def __str__(self):
        return f"EAPI with population size {self.population_size} and budget {self.budget}"

# BBOB test suite
def bboptest(func, bounds, budget):
    # Generate a set of noisy function evaluations
    evaluations = [func(x) for x in np.random.uniform(bounds[0], bounds[1], size=budget)]
    # Return the evaluations
    return evaluations

# Define a black box function
def func(x):
    return np.sin(x)

# Create an EAPI instance
api = EAPI(100, 10)

# Evaluate the function 100 times
evaluations = bboptest(func, (-5.0, 5.0), 100)

# Print the results
print(api)