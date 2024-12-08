import numpy as np

class DynamicIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def explore_phase(population, mutation_rates):
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def update_mutation_rates(mutation_rates, fitness_values):
            max_fitness = max(fitness_values)
            min_fitness = min(fitness_values)
            diff_fitness = max_fitness - min_fitness
            if diff_fitness != 0:
                adapt_ratio = 0.5 * np.log((max_fitness - fitness_values) / diff_fitness)
                mutation_rates = np.maximum(0.05, np.minimum(0.2, mutation_rates * np.exp(adapt_ratio)))
            return mutation_rates

        population = initialize_population()
        mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
        for _ in range(self.budget // 2):
            fitness_values = evaluate_population(population)
            mutation_rates = update_mutation_rates(mutation_rates, fitness_values)
            population = exploit_phase(explore_phase(population, mutation_rates))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution