import numpy as np

class DynamicIMPHS(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            mutation_factors = np.random.uniform(0.9, 1.1, self.dim)
            new_population = population + np.random.normal(0, mutation_rates * mutation_factors, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population
        return population