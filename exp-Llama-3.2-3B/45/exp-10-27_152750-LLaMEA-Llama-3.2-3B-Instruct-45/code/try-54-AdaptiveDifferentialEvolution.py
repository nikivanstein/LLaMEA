import numpy as np
import random
import time

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.adaptation_rate = 0.45
        self.fitness_history = []
        self.best_individuals = []
        self.differential_evolution_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.hybrid_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

    def __call__(self, func):
        start_time = time.time()
        for i in range(self.budget):
            # Initialize population with random points
            self.differential_evolution_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            self.hybrid_population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

            # Evaluate fitness for each individual
            fitness = np.array([func(individual) for individual in self.differential_evolution_population])

            # Differential evolution
            for j in range(self.population_size):
                # Select parents using tournament selection
                parents = []
                for _ in range(self.population_size):
                    tournament = random.sample(range(self.population_size), 3)
                    parents.append(self.differential_evolution_population[tournament[np.argmin(fitness[tournament])]])
                parents = np.array(parents)

                # Crossover
                offspring = []
                for _ in range(self.population_size):
                    if random.random() < self.crossover_probability:
                        parent1, parent2 = random.sample(parents, 2)
                        child = (parent1 + parent2) / 2
                        offspring.append(child)
                    else:
                        offspring.append(parents[np.random.randint(0, len(parents))])

                # Hybrid mutation
                for k in range(self.population_size):
                    if random.random() < self.mutation_probability:
                        mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                        offspring[k] += self.adaptation_rate * mutation

                # Replace worst individual
                offspring = np.sort(offspring, axis=0)
                self.differential_evolution_population = np.delete(self.differential_evolution_population, np.argmin(fitness), axis=0)
                self.differential_evolution_population = np.vstack((self.differential_evolution_population, offspring))

                # Update hybrid population
                self.hybrid_population = np.sort(self.hybrid_population, axis=0)

            # Evaluate fitness for each individual
            fitness = np.array([func(individual) for individual in self.differential_evolution_population])

            # Update fitness history and best individuals
            self.fitness_history.append(fitness)
            self.best_individuals.append(np.argmin(fitness))

        end_time = time.time()
        print(f"Optimization time: {end_time - start_time} seconds")
        return np.min(self.fitness_history), self.best_individuals[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

ade = AdaptiveDifferentialEvolution(budget=100, dim=10)
best_fitness, best_individual = ade(func)
print(f"Best fitness: {best_fitness}")
print(f"Best individual: {best_individual}")