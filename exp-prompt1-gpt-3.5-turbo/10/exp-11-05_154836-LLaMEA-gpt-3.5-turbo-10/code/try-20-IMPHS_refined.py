import numpy as np

class IMPHS_refined:
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
                for i in range(len(population)):
                    candidates = np.random.choice(np.delete(population, i, axis=0), 2, replace=False)
                    mutant = candidates[0] + 0.5 * (candidates[1] - population[i])
                    trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, population[i])
                    if func(trial) < func(population[i]):
                        population[i] = trial
            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution