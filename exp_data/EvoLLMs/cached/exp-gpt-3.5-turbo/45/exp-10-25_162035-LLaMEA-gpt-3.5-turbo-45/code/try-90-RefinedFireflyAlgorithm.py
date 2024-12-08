import numpy as np

class RefinedFireflyAlgorithm(EnhancedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def dynamic_mutation_rate(self, t, T):
        return 0.1 + 0.9 * (1 - t / T)
    
    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                rand_indices = np.random.choice(range(self.budget), 3, replace=False)
                mutation_rate = self.dynamic_mutation_rate(t, self.budget)
                mutant = self.differential_evolution(self.population[i], self.population[rand_indices[0]], self.population[rand_indices[1]], self.population[rand_indices[2]], f=mutation_rate)
                if func(mutant) < func(self.population[i]):
                    self.population[i] = mutant
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]