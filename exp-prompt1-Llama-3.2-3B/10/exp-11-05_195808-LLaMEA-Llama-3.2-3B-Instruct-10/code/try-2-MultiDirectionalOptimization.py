import numpy as np
import random

class MultiDirectionalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_score = -np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            fitness = np.array([func(solution) for solution in self.population])

            # Select the fittest individuals
            indices = np.argsort(fitness)
            self.population = self.population[indices]

            # Perform crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(indices[:self.population_size//2], 2)
                child = (parent1 + parent2) / 2
                child = child + np.random.uniform(-0.5, 0.5, self.dim)
                new_population.append(child)

            # Add the best solution to the new population
            new_population.append(self.best_solution)

            # Perform particle swarm optimization
            swarm = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            best_swarm = swarm.copy()
            for _ in range(10):
                for i in range(self.population_size):
                    r1, r2 = np.random.uniform(0, 1, 2)
                    x = self.population[i] + r1 * (best_swarm[i] - self.population[i])
                    y = self.population[i] + r2 * (self.best_solution - self.population[i])
                    best_swarm[i] = min(x, y, key=lambda x: func(x))

            # Update the population and the best solution
            self.population = np.array(new_population)
            self.best_solution = best_swarm[np.argmin([func(solution) for solution in best_swarm])]
            self.best_score = min(self.best_score, func(self.best_solution))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimization = MultiDirectionalOptimization(budget, dim)
for _ in range(100):
    optimization(func)
    print(optimization.best_score)