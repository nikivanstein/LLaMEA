import numpy as np

class AQIDE_ED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.Cr_base = 0.9  # Base crossover probability
        self.memory_F = []
        self.memory_Cr = []
        self.best_solution = None
        self.elite_percentage = 0.15  # Preserve top 15% elites

    def quantum_initialize(self):
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def dynamic_parameters(self):
        # Dynamically adjust F and Cr based on memory and diversity
        diversity = np.var(self.memory_F) if self.memory_F else 0.1
        self.F = self.F_base * (1 + diversity)
        self.Cr = self.Cr_base * (1 - diversity)

    def __call__(self, func):
        population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        self.best_solution = population[np.argmin(fitness)]

        while eval_count < self.budget:
            if len(self.memory_F) > 0 and len(self.memory_Cr) > 0:
                self.dynamic_parameters()

            new_population = np.copy(population)
            elite_count = int(self.pop_size * self.elite_percentage)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population[elite_indices] = population[elite_indices]

            for i in range(self.pop_size):
                if i in elite_indices:
                    continue

                x_best = population[np.argmin(fitness)]
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
                    self.memory_F.append(self.F)
                    self.memory_Cr.append(self.Cr)
                    if len(self.memory_F) > 100:
                        self.memory_F.pop(0)
                    if len(self.memory_Cr) > 100:
                        self.memory_Cr.pop(0)

                    if trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]