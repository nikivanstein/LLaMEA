import numpy as np

class AdaptiveMemoryHybridDE_RLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.func_evals = 0
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 2.0
        self.mutation_memory = np.full(self.pop_size, 0.8)
        self.crossover_memory = np.full(self.pop_size, 0.9)

    def adapt_parameters(self):
        for i in range(self.pop_size):
            if np.random.rand() < 0.3:
                self.mutation_memory[i] = np.clip(self.mutation_memory[i] + np.random.normal(0, 0.1), 0.5, 1.0)
                self.crossover_memory[i] = np.clip(self.crossover_memory[i] + np.random.normal(0, 0.1), 0.5, 1.0)

    def __call__(self, func):
        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            self.func_evals += 1
            if self.func_evals >= self.budget:
                return self.pop[np.argmin(self.fitness)]
        
        while self.func_evals < self.budget:
            self.adapt_parameters()
            
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.pop[np.argmin(self.fitness)]

                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.pop[indices]
                mutant = np.clip(a + self.mutation_memory[i] * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.crossover_memory[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                f_trial = func(trial)
                self.func_evals += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.pop[i] = trial

            if self.func_evals < self.budget:
                best_idx = np.argmin(self.fitness)
                for _ in range(self.dim):
                    random_idx = np.random.randint(self.pop_size)
                    local_search = np.clip(self.pop[random_idx] + np.random.uniform(-0.1, 0.1, self.dim), self.lb, self.ub)
                    f_local = func(local_search)
                    self.func_evals += 1
                    if f_local < self.fitness[random_idx]:
                        self.pop[random_idx] = local_search
                        self.fitness[random_idx] = f_local

        return self.pop[np.argmin(self.fitness)]