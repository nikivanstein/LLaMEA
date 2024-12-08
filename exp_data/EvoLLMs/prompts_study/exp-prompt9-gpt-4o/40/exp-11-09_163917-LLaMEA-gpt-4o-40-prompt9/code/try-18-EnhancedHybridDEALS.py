import numpy as np

class EnhancedHybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.mutation_factor = 0.85
        self.crossover_prob = 0.8
        self.local_search_prob = 0.25
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.global_search_prob = 0.1

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
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

                # Adaptive local search
                if np.random.rand() < self.local_search_prob:
                    self.adaptive_local_search(i, func)

                # Global search enhancement
                if np.random.rand() < self.global_search_prob:
                    self.global_search(func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def adaptive_local_search(self, index, func):
        step_size = 0.08 * (self.upper_bound - self.lower_bound)
        for _ in range(4):
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness

    def global_search(self, func):
        step_size = 0.25 * (self.upper_bound - self.lower_bound)
        for _ in range(2):
            if self.evaluations >= self.budget:
                break

            global_perturbation = np.random.uniform(-step_size, step_size, (self.population_size, self.dim))
            new_population = np.clip(self.population + global_perturbation, self.lower_bound, self.upper_bound)
            
            for j in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                new_fitness = func(new_population[j])
                self.evaluations += 1
                
                if new_fitness < self.fitness[j]:
                    self.population[j] = new_population[j]
                    self.fitness[j] = new_fitness