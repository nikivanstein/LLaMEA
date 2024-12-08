import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cross_prob = 0.9 + 0.1 / dim
        self.diff_weight = 0.5 + np.random.rand() * 0.5
        self.current_budget = 0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, population):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = population[idxs]
        progress_ratio = self.current_budget / self.budget
        adaptive_diff_weight = self.diff_weight * (1 - progress_ratio)
        mutant = a + adaptive_diff_weight * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cross_prob
        return np.where(crossover_mask, mutant, target)

    def _directional_search(self, current, func):
        direction = np.random.normal(size=self.dim)
        direction /= np.linalg.norm(direction)
        step_size = (self.upper_bound - self.lower_bound) * (0.1 + 0.1 * (1 - self.current_budget / self.budget))
        candidate = np.clip(current + step_size * direction, self.lower_bound, self.upper_bound)
        return candidate if func(candidate) < func(current) else current

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.current_budget += self.population_size

        while self.current_budget < self.budget:
            successes = 0
            for i in range(self.population_size):
                mutant = self._mutate(population)
                trial = self._crossover(population[i], mutant)
                trial = self._directional_search(trial, func)
                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    successes += 1

                if self.current_budget >= self.budget:
                    break

            if successes > 0:
                self.cross_prob = min(1.0, self.cross_prob + 0.01 * (successes / self.population_size))
                self.diff_weight *= (1 / (1 + successes))  # Change: Scale diff_weight inversely with success rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]