import numpy as np

class AdaptiveEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sigma = 0.1

    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.adaptive_mutate_population(func)
            self.evaluate_population(func)
        return self.best_solution

    def adaptive_mutate_population(self, func):
        for i, ind in enumerate(self.population):
            indices = [idx for idx in range(len(self.population)) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + 0.5 * (self.population[b] - self.population[c])
            new_ind = ind + self.sigma * mutant
            if func(new_ind) < func(ind):
                self.population[i] = new_ind
                self.sigma *= 1.1  # Increase mutation step size
            else:
                self.sigma /= 2.0  # Decrease mutation step size