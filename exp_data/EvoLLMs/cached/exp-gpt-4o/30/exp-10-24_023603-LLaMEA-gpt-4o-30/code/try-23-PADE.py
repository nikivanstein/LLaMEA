import numpy as np

class PADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.memory_size = 5
        self.cross_prob = 0.9
        self.F_base = 0.5
        self.epsilon = 0.01
        
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.memory = {
            "F": np.full(self.memory_size, self.F_base),
            "CR": np.full(self.memory_size, self.cross_prob)
        }
        self.memory_index = 0

    def update_memory(self, F, CR):
        self.memory["F"][self.memory_index] = F
        self.memory["CR"][self.memory_index] = CR
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def __call__(self, func):
        eval_count = 0
        best_fitness = np.inf
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                F_dynamic = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                if np.random.rand() < 0.5:
                    indices_extra = np.random.choice(self.population_size, 1, replace=False)
                    d = self.population[indices_extra]
                    mutant = np.clip(a + F_dynamic * (b - c) + F_dynamic * (d - a), self.lower_bound, self.upper_bound)
                else:
                    mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)

                CR_dynamic = np.random.choice(self.memory["CR"]) + np.random.rand() * self.epsilon
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if eval_count >= self.budget:
                    break

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        self.update_memory(F_dynamic, CR_dynamic)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]