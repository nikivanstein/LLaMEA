import numpy as np

class Enhanced_DE_SA_Metaheuristic_v4:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.alpha = budget, dim, 10, 0.5, 0.9
        self.F, self.T, self.population = 0.5, 1.0, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        best_solution, best_fitness = np.random.uniform(-5.0, 5.0, self.dim), func(np.zeros(self.dim))

        for _ in range(self.budget):
            new_population = []
            adapt_F = np.clip(self.F + 0.1 * np.random.randn(), 0.1, 0.9)
            mutants = self.population[np.random.choice(self.pop_size, (self.pop_size, 2), replace=False)]
            trials = self.population[:, None, :] + adapt_F * (mutants[:, 0, :] - mutants[:, 1, :])
            masks = np.random.rand(self.pop_size, self.dim) < self.CR
            trials[masks] = self.population[masks]

            new_fitness = np.array([func(trial) for trial in trials])
            improvements = new_fitness < best_fitness
            best_solution[improvements], best_fitness[improvements] = trials[improvements], new_fitness[improvements]

            exchange = new_fitness < func(self.population) + np.log(np.random.rand(self.pop_size)) * self.T
            self.population[exchange] = trials[exchange]
            new_population = np.where(exchange[:, None], trials, self.population)

            self.T = self.T * max(self.alpha, 0.1)

        return best_solution