import numpy as np

class HybridDERLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 5)
        self.mutation_factor = 0.85
        self.crossover_prob = 0.8
        self.local_search_prob = 0.25
        self.restart_interval = 0.2 * budget
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.best_solution = None
        self.best_fitness = float('inf')

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

                # Progressive local search adaptation
                if np.random.rand() < self.local_search_prob:
                    self.progressive_local_search(i, func)
            
            # Restart strategy
            if self.evaluations % self.restart_interval == 0:
                self.restart_population(func)

        self.best_solution = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        return self.best_solution, self.best_fitness

    def progressive_local_search(self, index, func):
        step_size = 0.05 * (self.upper_bound - self.lower_bound)
        for _ in range(7):
            if self.evaluations >= self.budget:
                break
            
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness

    def restart_population(self, func):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.population_size