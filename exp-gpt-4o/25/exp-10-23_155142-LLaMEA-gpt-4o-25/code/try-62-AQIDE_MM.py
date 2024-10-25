import numpy as np

class AQIDE_MM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 12 * dim  # Increased population size for diversity
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory_F1 = []
        self.memory_F2 = []
        self.memory_Cr1 = []
        self.memory_Cr2 = []
        self.best_solution = None
        self.elite_percentage = 0.20  # Increased elite preservation to 20%

    def quantum_initialize(self):
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def __call__(self, func):
        population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            if len(self.memory_F1) > 0 and len(self.memory_Cr1) > 0:
                self.F = np.median(self.memory_F1 + self.memory_F2)
                self.Cr = np.median(self.memory_Cr1 + self.memory_Cr2)

            new_population = np.copy(population)
            elite_count = int(self.pop_size * self.elite_percentage)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population[elite_indices] = population[elite_indices]

            for i in range(self.pop_size):
                if i in elite_indices:
                    continue

                x_best = population[np.argmin(fitness)] if self.best_solution is None else self.best_solution
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x_best + self.F * (x1 - x2 + x2 - x3), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if np.random.rand() > 0.5:
                        self.memory_F1.append(self.F)
                        self.memory_Cr1.append(self.Cr)
                    else:
                        self.memory_F2.append(self.F)
                        self.memory_Cr2.append(self.Cr)

                    if len(self.memory_F1) > 50:
                        self.memory_F1.pop(0)
                    if len(self.memory_F2) > 50:
                        self.memory_F2.pop(0)
                    if len(self.memory_Cr1) > 50:
                        self.memory_Cr1.pop(0)
                    if len(self.memory_Cr2) > 50:
                        self.memory_Cr2.pop(0)

                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]