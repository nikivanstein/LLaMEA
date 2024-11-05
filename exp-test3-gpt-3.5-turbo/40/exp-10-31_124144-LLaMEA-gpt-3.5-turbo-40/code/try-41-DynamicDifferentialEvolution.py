import numpy as np

class DynamicDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover rate
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for _ in range(self.budget - 1):
            target_vector = population[_]
            indices = list(range(self.budget))
            indices.remove(_)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = population[a] + self.F * (population[b] - population[c])
            crossover_points = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover_points, mutant_vector, target_vector)
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[_]:
                population[_] = trial_vector
                fitness[_] = trial_fitness
                if trial_fitness < fitness[best_idx]:
                    best_idx = _
                    best_solution = trial_vector
        return best_solution