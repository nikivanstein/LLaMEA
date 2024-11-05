# Description: Adaptive ADERM with dynamic population size to enhance exploration in early stages and fine-tune solutions later.
# Code: 
import numpy as np

class ADERM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.eval_count = 0

    def _mutate(self, idx):
        indices = [i for i in range(self.population.shape[0]) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 + np.random.uniform(-0.1, 0.1))
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.eval_count += self.population.shape[0]

        while self.eval_count < self.budget:
            current_size = self.population.shape[0]
            if self.eval_count > self.budget / 2:
                self.population = self.population[:current_size // 2]
                self.fitness = self.fitness[:current_size // 2]

            for i in range(self.population.shape[0]):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]