import numpy as np

class AdaptiveMemoryDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_factor = 0.5  # Memory influence factor

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        memory = best_solution.copy()  # Memory of the best solution

        while evals < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                dynamic_F = self.F * (1 - 0.5 * (evals / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lower_bound, self.upper_bound)

                # Adaptive Crossover Probability
                self.CR = 0.2 if evals > 0.7 * self.budget else 0.9

                crossover_mask = np.random.rand(self.dim) < self.CR
                crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                        memory = (1 - self.memory_factor) * memory + self.memory_factor * best_solution

                # Memory-based Local Search
                if evals % 50 == 0:
                    direction = np.random.uniform(-1.0, 1.0, self.dim)
                    adaptive_step = (0.05 + 0.95 * (evals / self.budget)) * (self.upper_bound - self.lower_bound)
                    local_trial = np.clip(memory + adaptive_step * direction, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[best_idx]:
                        best_solution = local_trial
                        fitness[best_idx] = local_fitness
                        memory = (1 - self.memory_factor) * memory + self.memory_factor * best_solution
        
        return best_solution