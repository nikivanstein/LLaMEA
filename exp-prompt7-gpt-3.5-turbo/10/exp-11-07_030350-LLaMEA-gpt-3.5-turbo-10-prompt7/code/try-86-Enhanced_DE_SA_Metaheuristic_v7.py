# import numpy as np

class Enhanced_DE_SA_Metaheuristic_v7:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.alpha, self.F, self.T = budget, dim, 10, 0.5, 0.9, 0.5, 1.0
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        best_solution, best_fitness = np.random.uniform(-5.0, 5.0, self.dim), func(np.zeros(self.dim))
        for _ in range(self.budget):
            adapt_F = np.clip(self.F + 0.1 * np.random.randn(), 0.1, 0.9)
            mutant_indices = np.random.choice(self.pop_size, (2, self.dim), replace=False)
            mutant = self.population[mutant_indices]
            trial = self.population + adapt_F * (mutant[0] - mutant[1])
            mask = np.random.rand(self.pop_size, self.dim) < self.CR
            trial = np.where(mask, trial, self.population)

            new_fitness = np.array([func(trial[i]) for i in range(self.pop_size)])
            improved_mask = new_fitness < best_fitness
            best_solution[improved_mask] = trial[improved_mask]
            best_fitness[improved_mask] = new_fitness[improved_mask]

            P = self.population
            update_mask = new_fitness < np.array([func(P[i]) for i in range(self.pop_size)]) \
                          + np.exp((np.array([func(P[i]) for i in range(self.pop_size)]) - new_fitness) / self.T) > np.random.rand(self.pop_size)
            self.population[update_mask] = trial[update_mask]

            self.T = self.T * max(self.alpha, 0.1)

        return best_solution