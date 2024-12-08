import numpy as np
import random
import operator

class EACM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.crossover_prob = 0.037037037037037035
        self.mutation_prob = 0.037037037037037035

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        population_size = 50
        population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(population_size, self.dim, 1))

        for _ in range(self.budget - self.f_evals):
            # Selection
            fitness = np.array([func(individual) for individual in population])
            selected_indices = np.argsort(fitness)[:10]

            # Crossover
            offspring = np.zeros((population_size, self.dim, 1))
            for i in range(population_size):
                if random.random() < self.crossover_prob:
                    parent1_index = selected_indices[i]
                    parent2_index = selected_indices[(i+1) % 10]
                    offspring[i] = (population[parent1_index] + population[parent2_index]) / 2

            # Mutation
            for i in range(population_size):
                if random.random() < self.mutation_prob:
                    mutation = np.random.uniform(-1, 1, size=self.dim)
                    offspring[i] += mutation

            # Bound checking
            offspring = np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])

            # Replacement
            population = np.vstack((population, offspring))

            # Update the best solution
            f_evals = func(offspring[0])
            x_best = offspring[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

eacm = EACM(budget=10, dim=2)
x_opt = eacm(func)
print(x_opt)