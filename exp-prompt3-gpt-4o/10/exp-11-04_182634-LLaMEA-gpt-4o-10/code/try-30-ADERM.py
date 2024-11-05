import numpy as np

class ADERM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 + np.random.uniform(-0.1, 0.1)) 
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        adaptive_CR = self.CR * (1 + np.random.uniform(-0.05, 0.05)) 
        crossover = np.random.rand(self.dim) < adaptive_CR
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.eval_count += self.population_size
        best_idx = np.argmin(self.fitness)
        best_fitness = self.fitness[best_idx]

        while self.eval_count < self.budget:
            dynamic_pop_size = max(4, int(self.population_size * (1 - self.eval_count / self.budget)))
            for i in range(dynamic_pop_size):
                if i != best_idx:
                    mutant = self._mutate(i)
                    trial = self._crossover(self.population[i], mutant)
                    trial_fitness = func(trial)
                    self.eval_count += 1

                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_idx = i
                            best_fitness = trial_fitness

                    if self.eval_count >= self.budget:
                        break

        return self.population[best_idx]