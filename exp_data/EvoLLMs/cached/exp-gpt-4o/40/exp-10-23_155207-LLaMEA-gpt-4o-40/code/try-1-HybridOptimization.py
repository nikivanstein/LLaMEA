import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = None
        self.fitness = None

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def differential_evolution(self, idx, func):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        
        trial = np.where(cross_points, mutant, self.population[idx])
        trial_fitness = func(trial)
        
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

    def adaptive_neighborhood_search(self, idx, func):
        step_size = (self.upper_bound - self.lower_bound) * 0.1
        for _ in range(5):
            neighbor = self.population[idx] + np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            if neighbor_fitness < self.fitness[idx]:
                self.population[idx] = neighbor
                self.fitness[idx] = neighbor_fitness

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        self.evaluate_population(func)
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                self.differential_evolution(i, func)
                evaluations += 1

                if evaluations >= self.budget:
                    break

                self.adaptive_neighborhood_search(i, func)
                evaluations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]