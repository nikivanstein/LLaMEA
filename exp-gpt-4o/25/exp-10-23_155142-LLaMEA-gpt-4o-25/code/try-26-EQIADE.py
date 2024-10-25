import numpy as np

class EQIADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F_min, self.F_max = 0.4, 0.9  # Differential weight bounds
        self.Cr = 0.9  # Crossover probability
        self.memory = []
        self.best_solution = None

    def hybrid_initialize(self):
        # Hybrid quantum-greedy initialization
        q_population = np.random.rand(self.pop_size, self.dim)
        greedy_factor = 0.7
        random_part = self.lower_bound + (self.upper_bound - self.lower_bound) * q_population
        greedy_part = np.clip(self.best_solution + greedy_factor * (self.upper_bound - self.lower_bound) * (np.random.rand(self.pop_size, self.dim) - 0.5), self.lower_bound, self.upper_bound)
        return np.where(np.random.rand(self.pop_size, 1) < 0.5, random_part, greedy_part)

    def __call__(self, func):
        # Initialize population
        if self.best_solution is None:
            population = self.hybrid_initialize()
        else:
            population = self.quantum_initialize()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Adaptive mutation strategy
            self.F = np.random.uniform(self.F_min, self.F_max)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Mutation with random individual consideration
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
                    if len(self.memory) > 50:  # Maintain limited memory size
                        self.memory.pop(0)

                    # Update best solution
                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]