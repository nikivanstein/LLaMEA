import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=50, F_base=0.7, CR_base=0.85):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
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
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                # Dynamic scaling factor with adaptive component
                F_dynamic = self.F_base + 0.1 * (np.random.rand() - 0.5) * (fitness[best_idx] / (fitness[i] + 1e-8))
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)
                
                # Adaptive crossover probability
                CR_dynamic = self.CR_base + 0.1 * (np.random.rand() - 0.5)
                cross_points = np.random.rand(self.dim) < CR_dynamic
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

                # Stochastic local search
                if evaluations % (self.pop_size // 3) == 0:
                    local_search_idx = np.random.choice(self.pop_size)
                    local_solution = population[local_search_idx]
                    perturbation = np.random.normal(0, 0.5, self.dim)
                    local_mutant = np.clip(local_solution + perturbation, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_mutant)
                    evaluations += 1
                    if local_fitness < fitness[local_search_idx]:
                        population[local_search_idx] = local_mutant
                        fitness[local_search_idx] = local_fitness
                        if local_fitness < fitness[best_idx]:
                            best_idx = local_search_idx
                            best = local_mutant.copy()

        return best