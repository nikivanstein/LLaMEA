import numpy as np

class MESADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.memory_size = 100  # Increased memory size for more adaptive learning
        self.memory = []

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Self-adaptive parameter control
            if len(self.memory) > 0:
                self.F = np.clip(np.mean([entry[0] for entry in self.memory]) + np.random.normal(0, 0.1), 0, 1)
                self.Cr = np.clip(np.mean([entry[1] for entry in self.memory]) + np.random.normal(0, 0.1), 0, 1)
            else:
                self.F = np.random.rand()
                self.Cr = np.random.rand()

            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((self.F, self.Cr))
                    if len(self.memory) > self.memory_size:
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]