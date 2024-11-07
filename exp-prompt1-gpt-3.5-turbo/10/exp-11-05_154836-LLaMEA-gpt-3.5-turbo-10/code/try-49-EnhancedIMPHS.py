import numpy as np

class EnhancedIMPHS(IMPHS):
    def exploit_phase(population, num_iterations=5):
        for _ in range(num_iterations):
            performance = evaluate_population(population)
            diversity = np.std(population, axis=0)
            diversity = diversity / np.max(diversity)
            mutation_rates = np.clip(0.1 * diversity, 0.01, 0.2)
            best_idx = np.argmin(performance)
            best_individual = population[best_idx]
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population
        return population