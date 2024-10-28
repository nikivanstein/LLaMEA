import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            self.population = np.concatenate((best_crowd_values, worst_population_values))

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform crossover and mutation with probability 0.5
            for i in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                    self.population[i] = child
                else:
                    mutated_population = self.population + np.random.normal(0, 1, self.population.shape)
                    self.population[i] = np.clip(mutated_population, -5.0, 5.0)

# Example usage:
def func(x):
    return np.sum(x**2)

bbo_benchmark = {
    "functions": [
        lambda x: np.sum(x**2),
        lambda x: np.sum(np.sin(x)**2),
        lambda x: np.sum(np.cos(x)**2),
        lambda x: np.sum(x**3),
        lambda x: np.sum(np.sin(x)**3),
        lambda x: np.sum(np.cos(x)**3),
        lambda x: np.sum(x**4),
        lambda x: np.sum(np.sin(x)**4),
        lambda x: np.sum(np.cos(x)**4),
        lambda x: np.sum(x**5),
        lambda x: np.sum(np.sin(x)**5),
        lambda x: np.sum(np.cos(x)**5),
        lambda x: np.sum(x**6),
        lambda x: np.sum(np.sin(x)**6),
        lambda x: np.sum(np.cos(x)**6),
        lambda x: np.sum(x**7),
        lambda x: np.sum(np.sin(x)**7),
        lambda x: np.sum(np.cos(x)**7),
        lambda x: np.sum(x**8),
        lambda x: np.sum(np.sin(x)**8),
        lambda x: np.sum(np.cos(x)**8),
        lambda x: np.sum(x**9),
        lambda x: np.sum(np.sin(x)**9),
        lambda x: np.sum(np.cos(x)**9),
    ],
    "dimensions": [10],
    "budget": 50,
    "iterations": 100,
    "population_size": 50,
    "crowd_size": 10,
    "crossover_probability": 0.5,
    "mutation_probability": 0.5,
}

# Initialize and run the CDEA algorithm
cdea = CDEA(bbo_benchmark["budget"], bbo_benchmark["dimensions"][0])
for i in range(bbo_benchmark["iterations"]):
    for func_name, func in bbo_benchmark["functions"].items():
        cdea(func)
        print(f"Iteration {i+1}: {func_name} - Best fitness: {np.min(cdea.population_values)}")