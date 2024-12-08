import numpy as np

class EMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.memory = []
        self.success_rate_memory = []

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive strategy selection based on memory
            if len(self.memory) > 0:
                self.F = np.mean([entry[0] for entry in self.memory])
                self.Cr = np.mean([entry[1] for entry in self.memory])
                success_rate = np.mean(self.success_rate_memory) if self.success_rate_memory else 0.5
                self.pop_size = max(4, int(10 * self.dim * success_rate))  # Dynamic population size

            new_population = np.copy(population)
            success_count = 0
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover with adaptive crossover control
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.memory.append((self.F, self.Cr))
                    success_count += 1
                    if len(self.memory) > 50:  # Maintain limited memory size
                        self.memory.pop(0)

                if eval_count >= self.budget:
                    break

            self.success_rate_memory.append(success_count / self.pop_size if self.pop_size else 0)
            if len(self.success_rate_memory) > 50:  # Maintain limited success rate memory size
                self.success_rate_memory.pop(0)

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]