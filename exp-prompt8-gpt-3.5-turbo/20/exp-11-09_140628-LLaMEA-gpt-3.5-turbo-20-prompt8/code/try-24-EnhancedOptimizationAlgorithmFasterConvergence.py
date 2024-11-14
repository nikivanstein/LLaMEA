import numpy as np

class EnhancedOptimizationAlgorithmFasterConvergence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(dim, 0.5)

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
                
                for i in range(self.dim):
                    mutation_rate = np.clip(self.mutation_rates[i] + np.random.normal(0, 0.1), 0.1, 0.9)
                    population[:, i] = 0.8*global_best[i] + 0.2*local_best[i] + mutation_rate * np.random.standard_normal(pop_size)
                
                fitness = np.array([func(individual) for individual in population])
        
        return best_individual