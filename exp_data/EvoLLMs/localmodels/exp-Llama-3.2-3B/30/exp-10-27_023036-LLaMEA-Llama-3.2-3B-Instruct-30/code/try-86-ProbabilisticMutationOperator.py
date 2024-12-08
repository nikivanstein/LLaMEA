import numpy as np
import random

class ProbabilisticMutationOperator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.score = -np.inf

    def __call__(self, func):
        # Initialize population with random points
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Evaluate population and update score
        self.score = max([func(x) for x in self.population])

        # Repeat for budget iterations
        for _ in range(self.budget):
            # Select best 20% of population
            self.population = sorted(self.population, key=lambda x: func(x))[:int(0.2 * len(self.population))]

            # Apply mutation operator
            mutated_population = []
            for i in range(len(self.population)):
                if random.random() < 0.3:
                    # Randomly select two parents and apply crossover
                    parent1, parent2 = random.sample(self.population, 2)
                    child = (parent1 + parent2) / 2
                    mutated_population.append(child)
                else:
                    mutated_population.append(self.population[i])

            # Update population and score
            self.population = mutated_population
            self.score = max([func(x) for x in self.population])

# Example usage:
def func(x):
    return np.sum(x**2)

operator = ProbabilisticMutationOperator(100, 10)
operator(func)
print(operator.score)