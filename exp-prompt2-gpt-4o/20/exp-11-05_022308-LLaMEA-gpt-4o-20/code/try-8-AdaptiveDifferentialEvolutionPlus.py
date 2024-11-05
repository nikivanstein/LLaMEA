import numpy as np

class AdaptiveDifferentialEvolutionPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Initial suggested population size
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Initial crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        dynamic_pop_size = self.pop_size

        while evals < self.budget:
            for i in range(dynamic_pop_size):
                # Mutation and crossover
                indices = np.random.choice(dynamic_pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                dynamic_cr = self.cr * (1 - (evals / self.budget))  # Adaptive crossover probability
                cross_points = np.random.rand(self.dim) < dynamic_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

            # Dynamic population resizing
            if evals % (self.budget // 10) == 0 and dynamic_pop_size > 4 * self.dim:
                dynamic_pop_size = int(dynamic_pop_size * 0.9)

        # Return best solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]