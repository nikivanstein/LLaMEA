import numpy as np
import random
import time

class MultiFidelityEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.fitness_history = []
        self.best_individuals = []
        self.fidelity_levels = 5
        self.fidelity_mapping = np.linspace(0, 1, self.fidelity_levels)

    def __call__(self, func):
        start_time = time.time()
        for i in range(self.budget):
            # Initialize population with random points
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

            # Evaluate fitness for each individual
            fitness = np.array([func(individual) for individual in population])

            # Evolutionary strategy
            for _ in range(100):  # 100 generations
                # Select parents using tournament selection
                parents = []
                for _ in range(self.population_size):
                    tournament = random.sample(range(self.population_size), 3)
                    parents.append(population[tournament[np.argmin(fitness[tournament])]])
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

                # Mutation
                for i in range(self.population_size):
                    if random.random() < self.mutation_probability:
                        mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                        offspring[i] += mutation

                # Replace worst individual
                offspring = np.sort(offspring, axis=0)
                population = np.delete(population, np.argmin(fitness), axis=0)
                population = np.vstack((population, offspring))

                # Evaluate fitness for each individual
                fitness = np.array([func(individual) for individual in population])

            # Update fitness history and best individuals
            self.fitness_history.append(fitness)
            self.best_individuals.append(np.argmin(fitness))

        # Fidelity-based selection
        fidelity_scores = np.random.uniform(0, 1, size=self.population_size)
        fidelity_mapping = np.dsplit(self.fidelity_mapping, self.fidelity_levels)
        for i in range(self.fidelity_levels):
            fidelity_mask = (fidelity_scores < fidelity_mapping[i]) & (fidelity_scores >= fidelity_mapping[i+1])
            fidelity_scores[fidelity_mask] = i
        population = population[fidelity_scores]

        # Update best individuals
        self.best_individuals = np.array([np.min(fitness) for fitness in self.fitness_history])

        end_time = time.time()
        print(f"Optimization time: {end_time - start_time} seconds")
        return np.min(self.fitness_history), self.best_individuals[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

mfs = MultiFidelityEvolutionStrategy(budget=100, dim=10)
best_fitness, best_individual = mfs(func)
print(f"Best fitness: {best_fitness}")
print(f"Best individual: {best_individual}")