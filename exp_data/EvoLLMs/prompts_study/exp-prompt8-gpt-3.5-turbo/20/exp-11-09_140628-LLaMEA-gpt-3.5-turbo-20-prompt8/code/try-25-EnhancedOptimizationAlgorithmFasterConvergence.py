import numpy as np

class EnhancedOptimizationAlgorithmFasterConvergence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.recombination_factors = np.full(dim, 0.5)

    def __call__(self, func):
        pop_sizes = np.random.randint(5, 15, self.budget)
        
        for pop_size in pop_sizes:
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            fitness = np.array([func(individual) for individual in population])
            
            for _ in range(self.budget):
                sorted_indices = np.argsort(fitness)
                best_individual = population[sorted_indices[0]]
                
                global_best = population[sorted_indices[0]]
                local_best = population[sorted_indices[1]]

                distances = np.linalg.norm(population.reshape((pop_size, 1, self.dim)) - population, axis=2)
                recombination_matrix = np.exp(-distances)
                
                for i in range(self.dim):
                    recombination_factor = np.clip(self.recombination_factors[i] + np.random.normal(0, 0.1), 0.1, 0.9)
                    population[:, i] = recombination_factor * global_best[i] + (1 - recombination_factor) * local_best[i] + np.sum(recombination_matrix * (population[:, i] - np.mean(population, axis=0)), axis=1)
                
                fitness = np.array([func(individual) for individual in population])
        
        return best_individual