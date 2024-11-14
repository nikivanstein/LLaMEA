import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Crossover probability
        self.population_size = max(4, int(10 * dim / 2))  # Dynamic population size
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def __call__(self, func):
        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1
            if self.used_budget >= self.budget:
                break

        while self.used_budget < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Enhanced Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.used_budget += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                
                if self.used_budget >= self.budget:
                    break

            # Adaptive mutation factor update and population resizing
            self.F = 0.5 + (0.3 * np.random.rand())
            # Resizing population dynamically based on performance
            best_idx = np.argmin(self.fitness)
            if self.used_budget < 0.5 * self.budget and i == best_idx:  # early success
                self.population_size = min(int(1.5 * self.population_size), 2 * self.dim)
                new_individuals = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size - len(self.population), self.dim))
                self.population = np.vstack((self.population, new_individuals))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.fitness = np.concatenate((self.fitness, new_fitness))

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]