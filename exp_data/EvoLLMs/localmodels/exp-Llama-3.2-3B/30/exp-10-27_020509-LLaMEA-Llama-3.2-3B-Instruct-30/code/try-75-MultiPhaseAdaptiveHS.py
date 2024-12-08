import numpy as np
import random

class MultiPhaseAdaptiveHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(budget * 0.2)
        self.bounds = [(-5.0, 5.0)] * dim
        self.best_solution = None
        self.best_fitness = float('inf')
        self.crossover_probabilities = np.zeros((self.population_size, self.population_size))

    def __call__(self, func):
        for phase in range(10):
            # Initialize population
            population = np.random.uniform(self.bounds[0], self.bounds[1], size=(self.population_size, self.dim))
            fitnesses = np.array([func(x) for x in population])

            # Selection
            selection_probabilities = fitnesses / np.sum(fitnesses)
            selected_indices = np.random.choice(self.population_size, size=self.population_size, p=selection_probabilities)
            selected_population = population[selected_indices]

            # Adaptation
            if phase < 5:
                # Adaptive population size
                new_population_size = int(self.population_size * 0.9 + random.uniform(0.1, 0.3))
                selected_indices = np.random.choice(self.population_size, size=new_population_size, p=selection_probabilities)
                selected_population = selected_population[selected_indices]
                self.population_size = new_population_size

            # Crossover
            for i in range(self.population_size):
                crossover_probabilities = np.random.uniform(0.5, 1.0, size=(self.population_size,))
                self.crossover_probabilities[i] = crossover_probabilities
                if np.random.rand() < 0.3:
                    j = np.random.randint(0, self.population_size)
                    parent1, parent2 = selected_population[i], selected_population[j]
                    for k in range(self.dim):
                        if np.random.rand() < self.crossover_probabilities[i, j]:
                            offspring = parent1.copy()
                            offspring[k] = parent2[k]
                            selected_population[i] = offspring

            # Mutation
            mutation_probabilities = np.random.uniform(0.1, 0.3, size=(self.population_size, self.dim))
            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() < mutation_probabilities[i, j]:
                        selected_population[i, j] += np.random.uniform(-1.0, 1.0)

            # Evaluation
            fitnesses = np.array([func(x) for x in selected_population])
            selected_indices = np.argsort(fitnesses)
            selected_population = selected_population[selected_indices]

            # Update best solution
            if np.min(fitnesses) < self.best_fitness:
                self.best_solution = selected_population[np.argmin(fitnesses)]
                self.best_fitness = np.min(fitnesses)

        return self.best_solution, self.best_fitness

# Example usage:
if __name__ == "__main__":
    def func(x):
        return np.sum(x**2)

    budget = 100
    dim = 10
    optimizer = MultiPhaseAdaptiveHS(budget, dim)
    best_solution, best_fitness = optimizer(func)
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)