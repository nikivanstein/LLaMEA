import numpy as np

class MADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 15 * dim  # Increased population size
        self.F = 0.5
        self.Cr = 0.9
        self.memory = []
        self.adaptive_memory_size = 100  # Adaptive memory size

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive parameter adjustment
            if len(self.memory) > 0:
                self.F = np.mean([entry[0] for entry in self.memory]) + np.random.normal(0, 0.1)
                self.Cr = np.mean([entry[1] for entry in self.memory]) + np.random.normal(0, 0.1)
            
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation with dynamic scaling factor
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                F_dynamic = self.F * (1 + np.random.uniform(-0.1, 0.1))
                mutant = np.clip(x1 + F_dynamic * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((F_dynamic, self.Cr))
                    if len(self.memory) > self.adaptive_memory_size:
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]