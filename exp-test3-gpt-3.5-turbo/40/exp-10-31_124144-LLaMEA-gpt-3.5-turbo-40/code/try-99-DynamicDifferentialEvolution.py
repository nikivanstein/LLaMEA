import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5 + 0.3 * np.random.rand()
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for _ in range(self.budget - 1):
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            r1, r2, r3 = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(population[idxs[0]] + self.f * (r1 - r2), -5.0, 5.0)
            cross_points = np.random.rand(self.dim) < 0.9
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[idxs[0]])
            f_val = func(trial)
            if f_val < fitness[idxs[0]]:
                fitness[idxs[0]] = f_val
                population[idxs[0]] = trial
        return best_solution