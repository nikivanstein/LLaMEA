import numpy as np

class HybridSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.mutation_factor = 0.9
        self.crossover_prob = 0.9
        self.local_search_prob = 0.3
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, float('inf'))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        self.personal_best_fitness = np.copy(self.fitness)
        global_best_idx = np.argmin(self.fitness)
        global_best = self.population[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.cognitive_coeff * r1 * (self.personal_best[i] - self.population[i])
                social = self.social_coeff * r2 * (global_best - self.population[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive + social
                
                # Particle swarm mutation
                mutant = self.population[i] + self.velocities[i]
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.personal_best_fitness[i]:
                        self.personal_best[i] = trial
                        self.personal_best_fitness[i] = trial_fitness
                        if trial_fitness < self.fitness[global_best_idx]:
                            global_best_idx = i
                            global_best = trial

                # Adaptive local exploration
                if np.random.rand() < self.local_search_prob * (1 - self.evaluations / self.budget):
                    self.adaptive_local_search(i, func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def adaptive_local_search(self, index, func):
        local_step = 0.2 * (self.upper_bound - self.lower_bound)
        neighbor = np.clip(self.population[index] + np.random.uniform(-local_step, local_step, self.dim), self.lower_bound, self.upper_bound)
        neighbor_fitness = func(neighbor)
        self.evaluations += 1

        if neighbor_fitness < self.fitness[index]:
            self.population[index] = neighbor
            self.fitness[index] = neighbor_fitness