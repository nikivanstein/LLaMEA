import numpy as np

class DynamicMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        mutation_scale = 0.5  # Initial mutation scale
        for _ in range(self.budget):
            for i in range(self.budget):
                candidate = population[i].copy()
                idxs = np.random.choice(np.delete(np.arange(self.budget), i, axis=0), 3, replace=False)
                mutant = population[idxs[0]] + mutation_scale * (population[idxs[1]] - population[idxs[2]])
                crossover = np.random.rand(self.dim) < 0.9
                candidate[crossover] = mutant[crossover]
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = candidate
                    
            if np.random.rand() < 0.1:  # Adaptive mutation update
                fitness_diff = np.max(fitness) - np.min(fitness)
                mutation_scale = mutation_scale * np.exp(1.0 / self.budget * (fitness_diff / (np.abs(fitness_diff) + 1e-6)))
        
        return best_solution