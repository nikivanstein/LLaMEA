import numpy as np

class DynamicIMPHS(IMPHS):
    def exploit_phase(population, num_iterations=5):
        mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
        for _ in range(num_iterations):
            best_idx = np.argmin(evaluate_population(population))
            best_individual = population[best_idx]
            mutation_rates = np.clip(mutation_rates + np.random.normal(0, 0.02, self.dim), 0.05, 0.2)
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population
        return population