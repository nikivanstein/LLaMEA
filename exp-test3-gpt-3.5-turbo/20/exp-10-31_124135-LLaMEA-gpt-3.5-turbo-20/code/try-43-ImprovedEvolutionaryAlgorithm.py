import numpy as np

class ImprovedEvolutionaryAlgorithm:
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
            
            # Crossover step to promote diversity and exploration
            crossover_population = np.array([self.crossover(population[i], population[np.random.randint(0, self.budget)]) for i in range(self.budget)])
            
            population = crossover_population
            evals += self.budget
        
        return best_solution

    def crossover(self, ind1, ind2):
        mask = np.random.choice([True, False], size=self.dim)
        new_ind = np.where(mask, ind1, ind2)
        return new_ind