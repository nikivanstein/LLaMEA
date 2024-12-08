import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()

        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                else:
                    indices = np.random.choice(self.pop_size // 2, 3, replace=False)
                x0, x1, x2 = population[indices]
                F_dynamic = self.F + 0.1 * (np.random.rand() - 0.5)
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                    
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial.copy()

                if evaluations >= self.budget:
                    break
                
                if evaluations % (self.pop_size // 2) == 0:
                    local_search_idx = np.argmin(fitness)
                    local_solution = population[local_search_idx]
                    local_mutant = np.clip(local_solution + 0.1 * np.random.randn(self.dim), self.lower_bound, self.upper_bound)
                    local_fitness = func(local_mutant)
                    evaluations += 1
                    if local_fitness < fitness[local_search_idx]:
                        population[local_search_idx] = local_mutant
                        fitness[local_search_idx] = local_fitness
                        if local_fitness < fitness[best_idx]:
                            best_idx = local_search_idx
                            best = local_mutant.copy()

        return best