import numpy as np

class Enhanced_DE_SA_Metaheuristic_v6:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.CR, self.alpha, self.F, self.T = budget, dim, 10, 0.5, 0.9, 0.5, 1.0
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def __call__(self, func):
        best_solution, best_fitness = np.random.uniform(-5.0, 5.0, self.dim), func(np.zeros(self.dim))

        for _ in range(self.budget):
            new_population = []
            for target in self.population:
                adapt_F = np.clip(self.F + 0.1 * np.random.randn(), 0.1, 0.9)
                mutant_indices = np.random.choice(self.pop_size, 2, replace=False)
                mutant_diff = self.population[mutant_indices[0]] - self.population[mutant_indices[1]]
                trial = target + adapt_F * mutant_diff
                mask = np.random.rand(self.dim) < self.CR
                trial[mask] = target[mask]

                new_fitness = func(trial)
                if new_fitness < best_fitness:
                    best_solution, best_fitness = trial, new_fitness

                P = target
                if new_fitness < func(target) or np.exp((func(P) - new_fitness) / self.T) > np.random.rand():
                    target = trial

                new_population.append(target)

            self.population, self.T = np.array(new_population), self.T * max(self.alpha, 0.1)

        return best_solution