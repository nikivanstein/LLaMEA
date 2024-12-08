import numpy as np

class DynamicIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def explore_phase(population):
            fitness_values = evaluate_population(population)
            min_fitness, max_fitness = np.min(fitness_values), np.max(fitness_values)
            norm_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness)
            mutation_rates = 0.05 + 0.15 * (1 - norm_fitness)
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution

dynamic_imphs = DynamicIMPHS(budget=1000, dim=10)