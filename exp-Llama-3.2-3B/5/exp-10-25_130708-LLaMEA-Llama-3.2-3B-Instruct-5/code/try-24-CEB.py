import numpy as np
from scipy.optimize import minimize

class CEB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness_history = []

    def __call__(self, func):
        # Evaluate initial population and store fitness history
        for x in self.population:
            f = func(x)
            self.fitness_history.append(f)

        # Main loop
        for _ in range(self.budget):
            # Select fittest individuals
            fittest_individuals = np.argsort(self.fitness_history)[-self.population_size:]
            fittest_population = self.population[fittest_individuals]

            # Select crossover points
            crossover_points = np.random.choice(self.dim, size=self.population_size, replace=False)

            # Create new population
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = fittest_population[i]
                parent2 = fittest_population[np.random.choice(self.population_size, 1)[0]]
                child = np.copy(parent1)
                for j in crossover_points[i]:
                    child[j] = parent2[j]
                new_population[i] = child

            # Evaluate new population
            for x in new_population:
                f = func(x)
                self.fitness_history.append(f)

            # Update population
            self.population = new_population

# Example usage
def func(x):
    return np.sum(x**2)

ceb = CEB(budget=100, dim=10)
ceb(func)