import numpy as np

class EMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F_base = 0.5
        self.Cr_base = 0.9
        self.memory = []
        self.memory_size = 100
        self.F_adapt = 0.7
        self.Cr_adapt = 0.7

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            if len(self.memory) > 0:
                self.F_base = np.clip(np.mean([entry[0] for entry in self.memory]) + np.random.normal(0, 0.1), 0.1, 0.9)
                self.Cr_base = np.clip(np.mean([entry[1] for entry in self.memory]) + np.random.normal(0, 0.1), 0.1, 0.9)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                F = self.F_base if np.random.rand() > 0.25 else self.F_adapt
                mutant = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                Cr = self.Cr_base if np.random.rand() > 0.25 else self.Cr_adapt
                crossover_mask = np.random.rand(self.dim) < Cr
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((F, Cr))
                    if len(self.memory) > self.memory_size:
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]