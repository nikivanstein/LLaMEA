import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        self.update_personal_best()
        self.update_global_best()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Particle Swarm Optimization dynamics
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.c1 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i]) +
                                    self.c2 * np.random.rand(self.dim) * (self.global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

                # Differential Evolution mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices)], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                
                self.update_personal_best(i)

            self.update_global_best()
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def update_personal_best(self, index=None):
        if index is not None:
            if self.fitness[index] < self.personal_best_fitness[index]:
                self.personal_best[index] = np.copy(self.population[index])
                self.personal_best_fitness[index] = self.fitness[index]
        else:
            for i in range(self.population_size):
                if self.fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best[i] = np.copy(self.population[i])
                    self.personal_best_fitness[i] = self.fitness[i]

    def update_global_best(self):
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.global_best_fitness:
            self.global_best = np.copy(self.population[best_idx])
            self.global_best_fitness = self.fitness[best_idx]