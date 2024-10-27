import numpy as np

class Hybrid_GA_SA_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.2
        self.initial_temperature = 100.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        temperature = self.initial_temperature

        for _ in range(self.budget):
            offspring = np.copy(population)
            for i in range(self.budget):
                if np.random.rand() < self.mutation_rate:
                    offspring[i] += np.random.normal(0, 1, size=self.dim)
            for i in range(self.budget):
                current_cost = func(population[i])
                new_cost = func(offspring[i])
                if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temperature):
                    population[i] = offspring[i]
                    if new_cost < func(best_solution):
                        best_solution = offspring[i]
            temperature *= 0.95

        return best_solution