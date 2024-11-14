import numpy as np

class HybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 5)
        self.mutation_factor = 0.85
        self.crossover_prob = 0.85
        self.local_search_prob = 0.4
        self.dynamic_step = 0.05
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

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

                # Dynamic adaptive local search
                if np.random.rand() < self.local_search_prob:
                    self.dynamic_adaptive_local_search(i, func)
            
            # Elitism: Retain the best solution found so far
            best_idx = np.argmin(self.fitness)
            best_solution, best_fitness = self.population[best_idx], self.fitness[best_idx]

        return best_solution, best_fitness

    def dynamic_adaptive_local_search(self, index, func):
        step_size = self.dynamic_step * (self.upper_bound - self.lower_bound)
        for _ in range(5):
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness
                self.dynamic_step *= 0.9  # Reduce step size dynamically to refine search