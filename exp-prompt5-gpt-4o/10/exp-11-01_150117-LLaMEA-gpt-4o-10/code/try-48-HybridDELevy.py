import numpy as np

class HybridDELevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v) ** (1 / beta)
        return step * L

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size

        while evaluations < self.budget:
            if evaluations % (self.budget // 10) == 0:  # Dynamic Population Resizing
                self.population_size = max(self.dim, self.population_size // 2)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

            # Periodic reinitialization for exploration
            if evaluations % (self.budget // 4) == 0:
                reinit_idx = np.random.choice(range(self.population_size), size=self.population_size // 4, replace=False)
                population[reinit_idx] = np.random.uniform(self.lower_bound, self.upper_bound, (len(reinit_idx), self.dim))
                for idx in reinit_idx:
                    fitness[idx] = func(population[idx])
                evaluations += len(reinit_idx)

            for i in range(self.population_size):
                # Mutation (Differential Evolution)
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                self.F = 0.3 + 0.4 * np.random.rand()  # Adaptive dynamic scaling
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                self.CR = np.clip(np.random.normal(0.9, 0.1), 0.5, 1.0)
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i][j] 
                                  for j in range(self.dim)])

                # Levy Flight for exploration
                if np.random.rand() < 0.5:
                    trial += self.levy_flight(0.01 * (self.upper_bound - self.lower_bound))
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]