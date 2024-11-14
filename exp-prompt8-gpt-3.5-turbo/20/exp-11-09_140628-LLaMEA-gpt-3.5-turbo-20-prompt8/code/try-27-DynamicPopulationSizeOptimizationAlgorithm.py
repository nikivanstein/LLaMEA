import numpy as np

class DynamicPopulationSizeOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full(dim, 0.5)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (10, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            best_individual = population[sorted_indices[0]]
            
            global_best = population[sorted_indices[0]]
            local_best = population[sorted_indices[1]]
            
            for i in range(self.dim):
                mutation_rate = np.clip(self.mutation_rates[i] + np.random.normal(0, 0.1), 0.1, 0.9)
                fitness_diff = (best_individual - population) @ (best_individual - population).T
                mutation_rate *= 1 + 0.1 * (fitness - fitness.min()) / (fitness.max() - fitness.min())
                population[:, i] = 0.8*global_best[i] + 0.2*local_best[i] + mutation_rate * np.random.standard_normal(10)
            
            fitness = np.array([func(individual) for individual in population])
            
            if np.random.uniform() < 0.5:
                pop_size = np.random.randint(5, 15)
                new_individuals = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
                new_fitness = np.array([func(individual) for individual in new_individuals])
                population = np.vstack([population, new_individuals])
                fitness = np.concatenate([fitness, new_fitness])

                if len(fitness) > 100:
                    worst_indices = np.argsort(fitness)[::-1][:10]
                    population = np.delete(population, worst_indices, axis=0)
                    fitness = np.delete(fitness, worst_indices)
        
        return best_individual