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

            # Perform probability-based crossover and mutation
            new_population = []
            for i in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                else:
                    child = self.population[i]
                new_population.append(child)
            self.population = np.array(new_population)

            # Perform Gaussian mutation
            self.population = self.population + np.random.normal(0, 1, self.population.shape)
            self.population = np.clip(self.population, -5.0, 5.0)

# Test the CDEA algorithm on the BBOB test suite
def evaluateBBOB(func, algorithm):
    scores = []
    for dim in range(1, 10):
        for noise in [0]:
            for seed in [0]:
                population = np.random.uniform(-5.0, 5.0, (algorithm.population_size, dim))
                crowd = np.random.uniform(-5.0, 5.0, (algorithm.crowd_size, dim))
                for _ in range(algorithm.budget):
                    population_values = np.array([func(x) for x in population])
                    crowd_values = np.array([func(x) for x in crowd])
                    best_indices = np.argsort(population_values)[:, -algorithm.crowd_size:]
                    best_crowd_values = crowd_values[best_indices]
                    worst_indices = np.argsort(population_values)[:, :algorithm.crowd_size]
                    worst_population_values = population_values[worst_indices]
                    population = np.concatenate((best_crowd_values, worst_population_values))
                    crowd = population[:algorithm.crowd_size]
                    population = np.array([child for child in population if np.any(np.abs(child) > 5.0)])
                    crowd = np.array([child for child in crowd if np.any(np.abs(child) > 5.0)])
                    population = population + np.random.normal(0, 1, population.shape)
                    population = np.clip(population, -5.0, 5.0)
                    crowd = crowd + np.random.normal(0, 1, crowd.shape)
                    crowd = np.clip(crowd, -5.0, 5.0)
                scores.append(np.mean(population_values))
    return scores

# Initialize the CDEA algorithm
cdea = CDEA(100, 10)

# Test the CDEA algorithm on the BBOB test suite
scores = evaluateBBOB(lambda x: x**2, cdea)
print(scores)