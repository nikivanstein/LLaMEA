import numpy as np
import random
import time

class EvolutionaryMultiObjectiveEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.fitness_history = []
        self.best_individuals = []
        self.line_search_probability = 0.45

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

                # Line search
                if random.random() < self.line_search_probability:
                    # Select a random individual
                    individual = population[np.random.randint(0, self.population_size)]
                    # Evaluate fitness at the individual's position
                    fitness_at_individual = func(individual)
                    # Evaluate fitness at the individual's position with a small perturbation
                    perturbed_individual = individual + np.random.uniform(-0.1, 0.1, size=self.dim)
                    fitness_at_perturbed_individual = func(perturbed_individual)
                    # Choose the better fitness value
                    if fitness_at_individual < fitness_at_perturbed_individual:
                        offspring[i] = individual
                    else:
                        offspring[i] = perturbed_individual

                # Replace worst individual
                offspring = np.sort(offspring, axis=0)
                population = np.delete(population, np.argmin(fitness), axis=0)
                population = np.vstack((population, offspring))

                # Evaluate fitness for each individual
                fitness = np.array([func(individual) for individual in population])

            # Update fitness history and best individuals
            self.fitness_history.append(fitness)
            self.best_individuals.append(np.argmin(fitness))

        end_time = time.time()
        print(f"Optimization time: {end_time - start_time} seconds")
        return np.min(self.fitness_history), self.best_individuals[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

es = EvolutionaryMultiObjectiveEvolutionStrategy(budget=100, dim=10)
best_fitness, best_individual = es(func)
print(f"Best fitness: {best_fitness}")
print(f"Best individual: {best_individual}")