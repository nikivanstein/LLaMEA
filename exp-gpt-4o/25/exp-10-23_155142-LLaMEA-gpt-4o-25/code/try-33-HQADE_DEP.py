import numpy as np

class HQADE_DEP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory = []
        self.best_solution = None
        self.elite_percentage = 0.1  # Preserve top 10% elites
        self.dynamic_adaptation_rate = 0.05  # Adaptation rate for elite percentage

    def quantum_initialize(self):
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def __call__(self, func):
        population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            if len(self.memory) > 0:
                self.F = np.mean([entry[0] for entry in self.memory])
                self.Cr = np.mean([entry[1] for entry in self.memory]) * (1 + self.dynamic_adaptation_rate * (np.random.rand() - 0.5))

            new_population = np.copy(population)
            elite_count = max(1, int(self.pop_size * self.elite_percentage))
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population[elite_indices] = population[elite_indices]

            for i in range(self.pop_size):
                if i in elite_indices:
                    continue

                x_best = population[np.argmin(fitness)] if self.best_solution is None else self.best_solution
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((self.F, self.Cr))
                    if len(self.memory) > 50:
                        self.memory.pop(0)

                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population
            self.elite_percentage = min(0.2, self.elite_percentage * (1 + self.dynamic_adaptation_rate * (np.random.rand() - 0.5)))

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]