import numpy as np

class Enhanced_DE_SA_Metaheuristic_v6:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.alpha, self.F, self.T = budget, dim, 10, 0.5, 0.9, 0.5, 1.0
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        best_solution, best_fitness = np.random.uniform(-5.0, 5.0, self.dim), func(np.zeros(self.dim))
        for _ in range(self.budget):
            adapt_F = np.clip(self.F + 0.1 * np.random.randn(), 0.1, 0.9)
            mutant_indices = np.random.choice(self.pop_size, (2, self.dim), replace=False)
            mutants = self.population[mutant_indices]
            trial = self.population + adapt_F * (mutants[0] - mutants[1])
            
            mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial = np.where(mask, trial, self.population)

            new_fitness = func(trial.T)
            improve_mask = new_fitness < best_fitness
            best_solution = np.where(improve_mask, trial, best_solution)
            best_fitness = np.where(improve_mask, new_fitness, best_fitness)

            P = self.population
            improve_mask = new_fitness < func(P.T).T
            accept_mask = np.exp((func(P.T) - new_fitness) / self.T) > np.random.rand(self.pop_size)
            target = np.where(improve_mask | accept_mask[:, np.newaxis], trial, P)

            self.population, self.T = target, self.T * max(self.alpha, 0.1)

        return best_solution