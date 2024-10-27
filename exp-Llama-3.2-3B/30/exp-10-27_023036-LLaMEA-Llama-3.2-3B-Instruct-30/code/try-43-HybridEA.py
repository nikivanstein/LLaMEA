import numpy as np
import random

class HybridEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.best_individual = self.population[0]

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness_values = func(self.population)

            # Selection
            parents = np.array([self.population[i] for i in np.argsort(self.fitness_values)[:-self.population_size]])
            parents = parents[:self.population_size]

            # Crossover
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                crossover_point = random.randint(0, self.dim - 1)
                offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Mutation
            for i in range(self.population_size):
                if random.random() < 0.1:
                    mutation_point = random.randint(0, self.dim - 1)
                    offspring[i, mutation_point] = random.uniform(-5.0, 5.0)

            # Replacement
            self.population = offspring
            self.fitness_values = func(self.population)

            # Update the best individual
            self.best_individual = np.max(self.population, axis=0)

    def refine_solution(self):
        # Select the best individual
        best_individual = self.best_individual

        # Refine the solution by changing individual lines with a probability of 0.3
        refined_solution = list(best_individual)
        for i in range(self.dim):
            if random.random() < 0.3:
                refined_solution[i] = random.uniform(-5.0, 5.0)
        refined_solution = np.array(refined_solution)

        return refined_solution

# Example usage
def bbb_function(x):
    return np.sum(x**2)

hybrid_ea = HybridEA(100, 10)
hybrid_ea(bbb_function)
print(hybrid_ea.refine_solution())