import numpy as np

class ADER:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.crossover_rate = 0.5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        population = initialize_population()
        fitness_values = np.array([func(individual) for individual in population])
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]

        for _ in range(self.budget):
            mutated_population = np.zeros_like(population)
            for i in range(self.population_size):
                candidates = np.random.choice(np.delete(np.arange(self.population_size), i, 0), 2, replace=False)
                donor_vector = population[candidates[0]] + self.crossover_rate * (population[candidates[1]] - population[i])
                mutated_population[i] = np.clip(donor_vector, self.lb, self.ub)

            mutated_fitness = np.array([func(individual) for individual in mutated_population])
            for i in range(self.population_size):
                if mutated_fitness[i] < fitness_values[i]:
                    population[i] = mutated_population[i]
                    fitness_values[i] = mutated_fitness[i]

            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < func(best_solution):
                best_solution = population[best_index]

        return best_solution