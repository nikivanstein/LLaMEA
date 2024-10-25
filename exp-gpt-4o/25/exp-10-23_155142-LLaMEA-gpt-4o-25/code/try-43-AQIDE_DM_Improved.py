import numpy as np

class AQIDE_DM_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = max(10 * dim, 50)  # Ensure a reasonable population size
        self.F = 0.7  # Differential weight adjustment
        self.Cr = 0.8  # Crossover probability adjustment
        self.memory_F = []
        self.memory_Cr = []
        self.best_solution = None
        self.elite_percentage = 0.2  # Increased elite percentage for more exploration

    def quantum_initialize(self):
        q_population = np.random.rand(self.pop_size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * q_population

    def __call__(self, func):
        population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            if len(self.memory_F) > 0 and len(self.memory_Cr) > 0:
                self.F = np.mean(self.memory_F)  # Use mean instead of median
                self.Cr = np.mean(self.memory_Cr)  # Use mean instead of median

            new_population = np.copy(population)
            elite_count = int(self.pop_size * self.elite_percentage)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population[elite_indices] = population[elite_indices]

            for i in range(self.pop_size):
                if i in elite_indices:
                    continue

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
                    self.memory_F.append(self.F)
                    self.memory_Cr.append(self.Cr)
                    if len(self.memory_F) > 100:
                        self.memory_F.pop(0)
                    if len(self.memory_Cr) > 100:
                        self.memory_Cr.pop(0)

                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]