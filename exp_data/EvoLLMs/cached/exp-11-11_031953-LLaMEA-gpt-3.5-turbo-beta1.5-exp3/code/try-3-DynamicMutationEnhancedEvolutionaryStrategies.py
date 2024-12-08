import numpy as np

class DynamicMutationEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_step = 0.1

    def mutate_population(self, func):
        for i, ind in enumerate(self.population):
            indices = [idx for idx in range(len(self.population)) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + 0.5 * (self.population[b] - self.population[c])
            new_ind = ind + self.mutation_step * mutant
            if func(new_ind) < func(ind):
                self.population[i] = new_ind
                self.mutation_step *= 1.1
            else:
                self.mutation_step *= 0.9