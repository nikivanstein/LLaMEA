import numpy as np

class ImprovedDynamicDEPP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor
        self.F_adapt = 0.5  # Adaptive scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
            for i in range(self.budget):
                idx = np.delete(np.arange(self.budget), i)
                a, b, c = population[np.random.choice(idx, 3, replace=False)]
                j_rand = np.random.randint(self.dim)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                f_trial = func(trial)

                if f_trial < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = f_trial

            # Adaptive scaling factor update
            self.F_min = max(0.2, self.F_min - self.F_adapt)
            self.F_max = min(0.8, self.F_max + self.F_adapt)

        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution