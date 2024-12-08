import numpy as np

class DynamicMutationDE:
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
            for i in range(self.budget):
                if i != best_idx:
                    r1, r2, r3 = np.random.choice(self.population.shape[0], 3, replace=False)
                    mutant_vector = self.population[r1] + 0.5 * (self.population[r2] - self.population[r3])
                    crossover_prob = np.random.rand(self.dim) < 0.8
                    trial_vector = np.where(crossover_prob, mutant_vector, self.population[i])
                    if func(trial_vector) < fitness[i]:
                        self.population[i] = trial_vector
        return self.population[np.argmin([func(ind) for ind in self.population])]