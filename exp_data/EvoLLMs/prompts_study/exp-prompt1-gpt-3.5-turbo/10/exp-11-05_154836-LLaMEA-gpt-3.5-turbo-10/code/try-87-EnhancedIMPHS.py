import numpy as np

class EnhancedIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_lower_bound = 0.05
        self.mutation_upper_bound = 0.2
        self.exploit_iterations = 5

    def __call__(self, func):
        def explore_phase(population):
            mutation_rates = np.random.uniform(self.mutation_lower_bound, self.mutation_upper_bound, self.dim)
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def exploit_phase(population):
            mutation_rates = np.random.uniform(self.mutation_lower_bound, self.mutation_upper_bound, self.dim)
            for _ in range(self.exploit_iterations):
                best_idx = np.argmin(evaluate_population(population))
                best_individual = population[best_idx]
                new_population = population + np.random.normal(0, mutation_rates, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
                population = new_population
            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution