import numpy as np

class DynamicIMPHS(IMPHS):
    def __call__(self, func):
        def explore_phase(population, mutation_rates):
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def exploit_phase(population, num_iterations=5):
            for _ in range(num_iterations):
                fitness_values = evaluate_population(population)
                best_idx = np.argmin(fitness_values)
                mutation_rates = np.clip(0.05 + 0.1 * (1 - fitness_values / np.max(fitness_values)), 0.05, 0.2)
                best_individual = population[best_idx]
                new_population = population + np.random.normal(0, mutation_rates, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
                population = new_population
            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
            population = exploit_phase(explore_phase(population, mutation_rates))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution