import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            fittest = population[sorted_indices[0]]
            pop_mean = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - pop_mean, axis=1))

            mutation_strength = 5.0 / (1.0 + diversity)

            mutated = population + mutation_strength * np.random.randn(self.budget, self.dim)
            mutated_fitness = [func(ind) for ind in mutated]
            
            for i in range(self.budget):
                if mutated_fitness[i] < fitness[i]:
                    population[i] = mutated[i]
                    fitness[i] = mutated_fitness[i]
                    mutation_strength *= 1.1  # Increase mutation strength if fitness improves

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution