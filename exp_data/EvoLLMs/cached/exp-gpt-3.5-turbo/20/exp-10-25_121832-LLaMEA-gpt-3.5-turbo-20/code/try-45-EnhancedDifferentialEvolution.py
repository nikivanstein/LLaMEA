import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_prob = 0.9
        self.scale_factor_min = 0.5
        self.scale_factor_max = 1.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i, x in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                j_rand = np.random.randint(self.dim)
                trial_vector = np.copy(x)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob or j == j_rand:
                        trial_vector[j] = a[j] + (self.scale_factor_min + np.random.rand() * (self.scale_factor_max - self.scale_factor_min)) * (b[j] - c[j])

                if func(trial_vector) < func(x):
                    population[i] = trial_vector

                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution