import numpy as np
from scipy.stats import levy

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.cr = 0.9
        self.f = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _mutate(self, population, target_idx):
        candidates = population[[idx for idx in range(self.population_size) if idx != target_idx]]
        a, b, c = candidates[np.random.choice(len(candidates), 3, replace=False)]
        return np.clip(a + self.f * levy.rvs(size=self.dim) * (b - c), self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutant):
        trial = np.copy(target)
        idxs = np.where(np.random.rand(self.dim) < self.cr)
        trial[idxs] = mutant[idxs]
        return trial

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            for i in range(self.population_size):
                target = population[i]
                mutant = self._mutate(population, i)
                trial = self._crossover(target, mutant)

                if func(trial) < func(target):
                    population[i] = trial
                    
                evals += 1
                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution