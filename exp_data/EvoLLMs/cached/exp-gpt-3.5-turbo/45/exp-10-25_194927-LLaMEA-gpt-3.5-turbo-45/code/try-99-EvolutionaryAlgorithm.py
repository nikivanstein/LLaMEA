import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_mutation_rate = 0.1
        self.final_mutation_rate = 0.01
        self.adaptive_mutation_rate = 1.0 - (self.final_mutation_rate / self.initial_mutation_rate)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]

        for _ in range(self.budget):
            mutated_population = population + np.random.normal(0, 1, (self.budget, self.dim)) * self.initial_mutation_rate
            mutated_population = np.clip(mutated_population, self.lower_bound, self.upper_bound)

            fitness_values = [func(ind) for ind in mutated_population]
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < func(best_solution):
                best_solution = mutated_population[best_idx]
            
            population = mutated_population
            self.initial_mutation_rate *= self.adaptive_mutation_rate

        return best_solution