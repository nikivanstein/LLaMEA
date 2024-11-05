import numpy as np

class EnhancedEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            # Mutation operator based on Gaussian distribution
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))

            # Introducing a crossover operator
            crossover_point = np.random.randint(0, self.dim, self.budget)
            crossover_population = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                crossover_population[i] = np.where(np.arange(self.dim) < crossover_point[i], population[i], mutated_population[i])
            population = crossover_population

            evals += self.budget
        
        return best_solution