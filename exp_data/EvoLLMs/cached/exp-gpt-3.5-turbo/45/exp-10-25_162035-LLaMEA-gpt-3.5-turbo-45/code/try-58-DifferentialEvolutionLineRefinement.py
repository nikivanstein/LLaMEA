import numpy as np

class DifferentialEvolutionLineRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.cr = 0.9
        self.f = 0.5
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

    def refine_lines(self, selected_solution):
        for line in range(self.dim):
            if np.random.rand() < 0.35:
                selected_solution[line] = np.random.uniform(self.lower_bound, self.upper_bound)
        return selected_solution

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                indices = np.delete(np.arange(self.budget), i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == np.random.randint(self.dim):
                        mutant[j] = self.population[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < func(self.population[i]):
                    self.population[i] = mutant
            best_index = np.argmin([func(ind) for ind in self.population])
            self.population[best_index] = self.refine_lines(self.population[best_index])

        return self.population[np.argmin([func(ind) for ind in self.population])]