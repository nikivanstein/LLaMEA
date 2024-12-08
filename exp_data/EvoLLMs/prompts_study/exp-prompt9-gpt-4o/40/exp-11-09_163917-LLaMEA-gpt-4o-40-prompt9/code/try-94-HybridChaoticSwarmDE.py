import numpy as np

class HybridChaoticSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.mutation_factor = 0.7
        self.crossover_prob = 0.9
        self.local_search_prob = 0.2
        self.tournament_size = 3
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, float('inf'))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.chaos_sequence = np.random.rand(self.population_size)

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size
        
        for i in range(self.population_size):
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best[i] = self.population[i]
                self.personal_best_fitness[i] = self.fitness[i]
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.population[best_idx]
            self.global_best_fitness = self.fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # PSO-like update
                inertia_weight = 0.5 + np.random.rand() / 2
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    2 * np.random.rand() * (self.personal_best[i] - self.population[i]) +
                                    2 * np.random.rand() * (self.global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)
                
                # DE mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(indices, self.tournament_size, replace=False)
                best_idx = min(chosen, key=lambda idx: self.fitness[idx])
                a, b, c = self.population[best_idx], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
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

                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = trial
                    self.personal_best_fitness[i] = trial_fitness

                if trial_fitness < self.global_best_fitness:
                    self.global_best = trial
                    self.global_best_fitness = trial_fitness

                # Chaotic local search
                if np.random.rand() < self.local_search_prob * (1 - self.evaluations / self.budget):
                    self.chaotic_local_search(i, func)
        
        return self.global_best, self.global_best_fitness

    def chaotic_local_search(self, index, func):
        chaos_factor = 0.2
        for _ in range(5):
            if self.evaluations >= self.budget:
                break

            self.chaos_sequence[index] = 4 * self.chaos_sequence[index] * (1 - self.chaos_sequence[index])
            step_size = chaos_factor * (self.upper_bound - self.lower_bound) * (0.5 - self.chaos_sequence[index])
            perturbation = np.random.normal(0, np.abs(step_size), self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness
                if neighbor_fitness < self.personal_best_fitness[index]:
                    self.personal_best[index] = neighbor
                    self.personal_best_fitness[index] = neighbor_fitness
                if neighbor_fitness < self.global_best_fitness:
                    self.global_best = neighbor
                    self.global_best_fitness = neighbor_fitness