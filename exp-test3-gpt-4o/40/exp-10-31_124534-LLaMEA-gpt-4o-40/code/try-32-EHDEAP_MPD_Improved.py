import numpy as np

class EHDEAP_MPD_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpopulations = 5
        self.subpop_size = (10 * dim) // self.subpopulations
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        populations = [self._initialize_population() for _ in range(self.subpopulations)]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        self.evaluations += self.subpop_size * self.subpopulations

        while self.evaluations < self.budget:
            for s, population in enumerate(populations):
                if self.evaluations >= self.budget:
                    break
                for i in range(self.subpop_size):
                    if self.evaluations >= self.budget:
                        break

                    # Dynamic Mutation Factor
                    self.mutation_factor = 0.5 + 0.3 * np.random.rand()
                    indices = list(range(self.subpop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self._mutate(population[a], population[b], population[c])

                    # Adaptive Crossover
                    self.crossover_prob = 0.9 * (1 - 0.5 * (self.evaluations / self.budget))
                    trial = self._crossover(population[i], mutant)

                    # Enhanced Local Search
                    refined_trial = self._local_search(trial, func)

                    # Selection
                    trial_fitness = func(refined_trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = refined_trial
                        fitness[s][i] = trial_fitness

                # Dynamic Subpopulation Size
                if (self.evaluations / self.budget) > 0.5 and (s == 0):
                    self._adjust_subpopulations(populations, fitness)

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _mutate(self, a, b, c):
        mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _local_search(self, trial, func):
        step_size = 0.01 * (0.5 + 0.5 * np.random.rand())
        local_best = trial
        local_best_fitness = func(local_best)
        for _ in range(5):
            candidate = local_best + step_size * np.random.normal(0, 1, self.dim)
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < local_best_fitness:
                local_best = candidate
                local_best_fitness = candidate_fitness
        return local_best
    
    def _adjust_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.min() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            populations[sorted_indices[i]] = self._initialize_population()
            fitness[sorted_indices[i]] = np.array([np.inf] * self.subpop_size)