import numpy as np

class DifferentialEvolution:
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

            # Differential Evolution crossover operator
            mutated_population = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutated_population[i] = np.clip(a + 0.5 * (b - c), -5.0, 5.0)
            population = mutated_population
            evals += self.budget
        
        return best_solution