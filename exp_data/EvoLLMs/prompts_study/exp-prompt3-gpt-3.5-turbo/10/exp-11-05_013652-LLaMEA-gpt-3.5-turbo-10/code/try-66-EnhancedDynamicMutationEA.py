import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rate = np.random.uniform(0.01, 0.1)
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
            else:
                # Adjust population size dynamically based on success rate
                success_rate = sum([1 for f in fitness if f < func(best_solution)]) / len(fitness)
                if success_rate > 0.5:
                    self.population = np.vstack((self.population, np.random.uniform(-5.0, 5.0, (1, self.dim))))
                elif success_rate < 0.2 and len(self.population) > 1:
                    worst_idx = np.argmax(fitness)
                    self.population = np.delete(self.population, worst_idx, axis=0)
        return self.population[np.argmin([func(ind) for ind in self.population])]