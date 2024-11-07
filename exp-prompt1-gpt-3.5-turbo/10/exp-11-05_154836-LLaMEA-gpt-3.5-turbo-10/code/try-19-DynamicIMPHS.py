import numpy as np

class DynamicIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def explore_phase(population, mutation_rates):
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def exploit_phase(population, mutation_rates, num_iterations=5):
            for _ in range(num_iterations):
                best_idx = np.argmin(evaluate_population(population))
                best_individual = population[best_idx]
                new_population = population + np.random.uniform(-mutation_rates, mutation_rates, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
                population = new_population
            return population

        population = initialize_population()
        mutation_rates = np.full(self.dim, 0.1)  # Initial mutation rates
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population, mutation_rates), mutation_rates)
            mutation_rates = np.maximum(mutation_rates * 0.95, 0.01)  # Update mutation rates adaptively
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution