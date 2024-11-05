import numpy as np

class RefinedEvolutionaryAlgorithm:
    def __init__(self, budget, dim, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.elitism_rate = elitism_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Mutation operator based on Gaussian distribution
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            
            # Introducing elitism by preserving a fraction of top solutions
            num_elites = int(self.budget * self.elitism_rate)
            elite_indices = np.argsort(fitness)[:num_elites]
            mutated_population[elite_indices] = population[elite_indices]
            
            population = mutated_population
            evals += self.budget
        
        return best_solution