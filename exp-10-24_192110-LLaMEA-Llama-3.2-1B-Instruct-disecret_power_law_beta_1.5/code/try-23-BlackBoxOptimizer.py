import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            if self.budget > 0:
                self.population.append((solution, func(solution)))

        # Evolve population
        for _ in range(100):
            # Select fittest solutions
            fittest = sorted(self.population, key=lambda x: x[1], reverse=True)[:self.budget]

            # Select next generation
            next_gen = []
            for _ in range(self.budget):
                # Select two parents at random
                parent1, parent2 = random.sample(fittest, 2)
                # Create child by linear interpolation between parents
                child = (parent1[0] + (parent2[0] - parent1[0]) * random.random(), parent1[1] + (parent2[1] - parent1[1]) * random.random())
                # Ensure child is within bounds
                child = np.clip(child, -5.0, 5.0)
                next_gen.append(child)

            # Replace worst fittest solutions with child solutions
            worst_index = fittest.index(min(fittest, key=lambda x: x[1]))
            fittest[worst_index] = next_gen[worst_index]

            # Remove worst fittest solutions
            fittest = [solution for solution, _ in fittest]

            # Update population
            self.population = fittest

# Example usage
optimizer = BlackBoxOptimizer(budget=10, dim=5)
optimizer(func=lambda x: x**2)  # Evaluate example function
print(optimizer.population)