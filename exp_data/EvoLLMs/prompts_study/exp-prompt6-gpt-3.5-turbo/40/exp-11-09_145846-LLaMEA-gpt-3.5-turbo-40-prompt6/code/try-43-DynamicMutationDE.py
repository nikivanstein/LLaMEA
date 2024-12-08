import numpy as np

class DynamicMutationDE:
    def __init__(self, budget, dim, F_min=0.2, F_max=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR = CR

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        F = np.full(self.budget, self.F_min)  # Initialize mutation factor
        
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + F[i] * (population[best_index] - population[a]) + F[i] * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial
                        # Adjust mutation factor based on improvement
                        F[i] = min(self.F_max, F[i] * 1.2) if f_trial < fitness[i] else max(self.F_min, F[i] * 0.8)

        return best_solution