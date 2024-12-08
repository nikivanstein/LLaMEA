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
        self.phase = 0

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
            crossover_probabilities = np.random.uniform(0.5, 1.0, size=(self.population_size, self.population_size))
            crossover_indices = np.where(crossover_probabilities > 0.5)[0]
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = selected_population[crossover_indices[i][0]], selected_population[crossover_indices[i][1]]
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        offspring[i, j] = parent1[j]
                    else:
                        offspring[i, j] = parent2[j]

            # Mutation
            mutation_probabilities = np.random.uniform(0.1, 0.3, size=(self.population_size, self.dim))
            mutation_indices = np.where(mutation_probabilities > 0.5)[0]
            for i in range(self.population_size):
                for j in range(self.dim):
                    if np.random.rand() < mutation_probabilities[i, j]:
                        offspring[i, j] += np.random.uniform(-1.0, 1.0)

            # Evaluation
            fitnesses = np.array([func(x) for x in offspring])
            selected_indices = np.argsort(fitnesses)
            selected_offspring = offspring[selected_indices]

            # Update best solution
            if np.min(fitnesses) < self.best_fitness:
                self.best_solution = selected_offspring[np.argmin(fitnesses)]
                self.best_fitness = np.min(fitnesses)

            # Refine the strategy
            if self.phase < 5:
                refined_population = np.zeros((self.population_size, self.dim))
                for i in range(self.population_size):
                    refined_individual = self.best_solution[i]
                    for j in range(self.dim):
                        if np.random.rand() < 0.3:
                            refined_individual[j] += np.random.uniform(-1.0, 1.0)
                    refined_population[i] = refined_individual
                refined_fitnesses = np.array([func(x) for x in refined_population])
                refined_indices = np.argsort(refined_fitnesses)
                refined_offspring = refined_population[refined_indices]
                fitnesses = refined_fitnesses
                selected_indices = refined_indices

            self.phase += 1

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