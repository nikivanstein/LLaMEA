import numpy as np

class EMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F_min, self.F_max = 0.4, 0.9  # Dynamic differential weight range
        self.Cr_min, self.Cr_max = 0.7, 0.95  # Dynamic crossover probability range
        self.memory = []

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive parameter strategy based on memory
            if len(self.memory) > 0:
                self.F = np.clip(np.mean([entry[0] for entry in self.memory]), self.F_min, self.F_max)
                self.Cr = np.clip(np.mean([entry[1] for entry in self.memory]), self.Cr_min, self.Cr_max)
            else:
                self.F = (self.F_min + self.F_max) / 2
                self.Cr = (self.Cr_min + self.Cr_max) / 2
            
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                F_dynamic = np.random.uniform(self.F_min, self.F_max)
                mutant = np.clip(x1 + F_dynamic * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                j_rand = np.random.randint(self.dim)
                crossover_mask[j_rand] = True  # Ensure at least one dimension is swapped
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((F_dynamic, self.Cr))
                    if len(self.memory) > 50:  # Maintain limited memory size
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]