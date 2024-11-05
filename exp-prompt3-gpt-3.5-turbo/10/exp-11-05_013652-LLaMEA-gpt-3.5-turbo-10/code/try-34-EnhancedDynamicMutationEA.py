import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim, num_populations=5):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.populations = [np.random.uniform(-5.0, 5.0, (budget, dim)) for _ in range(num_populations)]
    
    def __call__(self, func):
        mutation_rates = np.random.uniform(0.01, 0.1, self.num_populations)
        for _ in range(self.budget):
            for i in range(self.num_populations):
                population = self.populations[i]
                fitness = [func(ind) for ind in population]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
                mutation = np.random.randn(self.dim) * mutation_rates[i]
                new_solution = best_solution + mutation
                if func(new_solution) < fitness[best_idx]:
                    population[best_idx] = new_solution
        all_solutions = np.vstack([pop for pop in self.populations])
        return all_solutions[np.argmin([func(ind) for ind in all_solutions])]