import numpy as np

class ImprovedENHMPHS(ENHMPHS):
    def exploit_phase(self, population, num_iterations=5):
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            mutation_scale = np.clip(0.1 / np.sqrt(np.mean((population - best_individual) ** 2, axis=0)), 0.05, 0.2)
            new_population = population + np.random.uniform(-mutation_scale, mutation_scale, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population
        return population