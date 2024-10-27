import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.CR = 0.8
        self.F = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _mutate(self, pop, target_idx, best_idx):
        idxs = [idx for idx in range(self.pop_size) if idx not in (target_idx, best_idx)]
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
        return np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutated):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_points, mutated, target)
        return trial

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            for i in range(self.pop_size):
                target_idx = i
                best_idx = np.argmin(self._get_fitness(population, func))
                mutated = self._mutate(population, target_idx, best_idx)
                trial = self._crossover(population[target_idx], mutated)

                if func(trial) < func(population[target_idx]):
                    population[target_idx] = trial

                evals += 1

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution