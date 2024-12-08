import numpy as np

class DifferentialEvolutionFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def differential_evolution(self, current, target, other, f=0.5):
        mutant = current + f * (target - other)
        return mutant
    
    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                rand_indices = np.random.choice(range(self.budget), 3, replace=False)
                mutant = self.differential_evolution(self.population[i], self.population[rand_indices[0]], self.population[rand_indices[1]], self.population[rand_indices[2]])
                if func(mutant) < func(self.population[i]):
                    self.population[i] = mutant
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]