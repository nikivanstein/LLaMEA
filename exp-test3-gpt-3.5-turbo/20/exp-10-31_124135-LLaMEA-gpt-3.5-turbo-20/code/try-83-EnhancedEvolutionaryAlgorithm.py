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
            
            # Crossover to promote diversity
            crossover_point = np.random.randint(1, self.dim)
            for i in range(self.budget):
                if i != best_idx:
                    crossover_mask = np.random.choice([0, 1], size=self.dim)
                    mutated_population[i] = crossover_mask * mutated_population[i] + (1 - crossover_mask) * best_solution
            population = mutated_population
            evals += self.budget

        return best_solution