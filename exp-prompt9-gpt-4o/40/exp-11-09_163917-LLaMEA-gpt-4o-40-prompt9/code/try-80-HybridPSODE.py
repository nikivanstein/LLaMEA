import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.mutation_factor = 0.8
        self.crossover_prob = 0.85
        self.local_search_prob = 0.25
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        self.personal_best_fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.population[best_idx]
            self.global_best_fitness = self.personal_best_fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocity[i] = (self.w * self.velocity[i] +
                                   self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                   self.c2 * r2 * (self.global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

                trial = self.differential_evolution_strategy(i)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = trial
                    self.personal_best_fitness[i] = trial_fitness

                if self.personal_best_fitness[i] < self.global_best_fitness:
                    self.global_best = self.personal_best[i]
                    self.global_best_fitness = self.personal_best_fitness[i]

                if np.random.rand() < self.local_search_prob * (1 - self.evaluations / self.budget):
                    self.local_search(i, func)

        return self.global_best, self.global_best_fitness

    def differential_evolution_strategy(self, index):
        indices = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(indices)], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        
        cross_points = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        
        trial = np.where(cross_points, mutant, self.population[index])
        return trial

    def local_search(self, index, func):
        step_size = 0.05 * (self.upper_bound - self.lower_bound)
        for _ in range(3):
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.personal_best_fitness[index]:
                self.personal_best[index] = neighbor
                self.personal_best_fitness[index] = neighbor_fitness
                if neighbor_fitness < self.global_best_fitness:
                    self.global_best = neighbor
                    self.global_best_fitness = neighbor_fitness