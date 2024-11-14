import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.mutation_factor = 0.9
        self.crossover_prob = 0.8
        self.learning_rate = 0.2
        self.adaptive_window = 0.3
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        global_best_idx = np.argmin(self.fitness)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(indices, 3, replace=False)
                a, b, c = self.population[chosen[0]], self.population[chosen[1]], self.population[chosen[2]]
                
                mutation_factor_dynamic = self.mutation_factor * (1 - self.evaluations / self.budget)
                mutant = np.clip(a + mutation_factor_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.fitness[global_best_idx]:
                        global_best_idx = i

                if np.random.rand() < self.learning_rate:
                    self.dynamic_local_search(i, func, global_best_idx)

        return self.population[global_best_idx], self.fitness[global_best_idx]

    def dynamic_local_search(self, index, func, global_best_idx):
        exploration_factor = 0.5
        global_best = self.population[global_best_idx]
        for _ in range(2):
            if self.evaluations >= self.budget:
                break

            direction = np.random.uniform(-exploration_factor, exploration_factor, self.dim)
            candidate = np.clip(self.population[index] + direction * (global_best - self.population[index]), self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            self.evaluations += 1
            
            if candidate_fitness < self.fitness[index]:
                self.population[index] = candidate
                self.fitness[index] = candidate_fitness