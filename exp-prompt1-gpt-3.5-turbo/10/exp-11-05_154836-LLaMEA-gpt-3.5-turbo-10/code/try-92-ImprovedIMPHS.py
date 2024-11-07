import numpy as np

class ImprovedIMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def explore_phase(population):
            mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def exploit_phase(population, num_iterations=5):
            for _ in range(num_iterations):
                best_idx = np.argmin(evaluate_population(population))
                best_individual = population[best_idx]
                new_population = population + np.random.uniform(-0.1, 0.1, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

                # Differential Evolution Step
                crossover_rate = 0.5
                for idx, ind in enumerate(population):
                    a, b, c = population[np.random.choice(len(population), 3, replace=False)]
                    trial = np.where(np.random.uniform(0, 1, self.dim) < crossover_rate, a + 0.5 * (b - c), ind)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)
                    if func(trial) < func(ind):
                        population[idx] = trial

            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution