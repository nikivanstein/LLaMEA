import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer: Adaptive Black Box Evolutionary Optimization"

    def adaptive_black_box_evolutionary_optimization(self, func, dim, budget):
        # Initialize the population with random values
        population = np.random.uniform(-5.0, 5.0, (self.dim,)) + 1.0

        # Evolve the population using the Adaptive Black Box Evolutionary Optimization algorithm
        for _ in range(1000):
            # Select the fittest individuals
            fittest = np.argmax(np.abs(self.func_values))
            population[fittest] = func(population[fittest])

            # Evolve the population using the Adaptive Black Box Evolutionary Optimization algorithm
            for _ in range(self.budget):
                # Select two parents using tournament selection
                parent1, parent2 = np.random.choice(population, size=2, replace=False)

                # Select the child using crossover
                child = func(parent1 + parent2)

                # Mutate the child using mutation
                if random.random() < 0.1:
                    child = func(child) + random.uniform(-1.0, 1.0)

                # Replace the parents with the child
                population[fittest] = child

        return population

# Test the algorithm
def test_adaptive_black_box_evolutionary_optimization(func, dim, budget):
    abeo = AdaptiveBlackBoxOptimizer(budget, dim)
    population = abeo.adaptive_black_box_evolutionary_optimization(func, dim, budget)
    return abeo

# Run the test
func = lambda x: x**2
dim = 10
budget = 100
population = test_adaptive_black_box_evolutionary_optimization(func, dim, budget)
print("Population:", population)