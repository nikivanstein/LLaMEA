import numpy as np

class NovelOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9 # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def opposition_based_learning(self, individual):
        return self.lower_bound + self.upper_bound - individual

    def differential_evolution_step(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), 
                         self.lower_bound, self.upper_bound)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[idx])
        return trial

    def __call__(self, func):
        evals = 0

        # Initial evaluation of the population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
        
        # Main optimization loop
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Evolution Step
                trial = self.differential_evolution_step(i)
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Opposition-based learning
                if evals < self.budget:
                    opposition = self.opposition_based_learning(trial)
                    opp_fitness = func(opposition)
                    evals += 1
                    if opp_fitness < self.fitness[i]:
                        self.population[i] = opposition
                        self.fitness[i] = opp_fitness

                if evals >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]