import numpy as np

class EnhancedIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def exploit_phase(population, num_iterations=5):
            for _ in range(num_iterations):
                best_idx = np.argmin(evaluate_population(population))
                best_individual = population[best_idx]
                new_population = population + np.random.uniform(-0.1, 0.1, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
                population = new_population
            new_population_size = int(1.2 * self.budget / (np.min(evaluate_population(population)) + 1))
            population = np.vstack([population, np.random.uniform(self.lower_bound, self.upper_bound, (new_population_size, self.dim))])
            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution