import numpy as np

class AdaptiveIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def explore_phase(population):
            mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
            performance = evaluate_population(population)
            for i in range(self.dim):
                mutation_rates[i] *= np.mean(np.abs(population[:, i] - np.mean(population[:, i]))) / (np.std(population[:, i]) + 1e-6)
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution