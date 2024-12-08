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
            fitness_ranking = np.argsort(np.argsort(fitness))  # Introducing fitness ranking
            
            for i in range(self.budget):
                mutation_factor = 1.0 / (1.0 + fitness_ranking[i])  # Dynamic mutation strategy
                mutated_individual = population[i] + mutation_strength * mutation_factor * np.random.randn(self.dim)
                mutated_fitness = func(mutated_individual)
                
                if mutated_fitness < fitness[i]:
                    population[i] = mutated_individual
                    fitness[i] = mutated_fitness

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        return best_solution