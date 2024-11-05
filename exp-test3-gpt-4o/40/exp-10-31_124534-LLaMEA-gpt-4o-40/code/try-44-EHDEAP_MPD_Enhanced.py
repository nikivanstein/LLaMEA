import numpy as np

class EHDEAP_MPD_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpopulations = 5
        self.subpop_size = (10 * dim) // self.subpopulations  # Changed from 12 to 10
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.85  # Changed from 0.8 to 0.85
        self.base_crossover_prob = 0.8  # Changed from 0.9 to 0.8
        self.crossover_decay_rate = 0.05  # Changed from 0.1 to 0.05
        self.evaluations = 0
        self.elite_fraction = 0.25  # Changed from 0.3 to 0.25
        self.merging_interval = 150  # Changed from 100 to 150

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

                    elite_indices = np.argsort(fitness[s])[:int(self.elite_fraction * self.subpop_size)]
                    elite = population[np.random.choice(elite_indices)]
                    indices = list(range(self.subpop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self._adaptive_mutate(population[a], elite, population[c])

                    self.crossover_prob = self.base_crossover_prob - (self.crossover_decay_rate * (self.evaluations / self.budget))
                    trial = self._crossover(population[i], mutant)

                    refined_trial = self._dynamic_local_search(trial, func)

                    trial_fitness = func(refined_trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = refined_trial
                        fitness[s][i] = trial_fitness
                
                if self.evaluations % self.merging_interval == 0:
                    self._merge_subpopulations(populations, fitness)

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _adaptive_mutate(self, a, b, c):
        # Introduced a new mutation strategy with randomness
        rand_factor = np.random.rand()
        mutant = np.clip(a + self.mutation_factor * (b - c) * rand_factor, self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _dynamic_local_search(self, trial, func):
        # Introduced dynamic step size adjustment
        alpha = 0.01 + 0.01 * (self.evaluations / self.budget)
        step_size = alpha * np.random.normal(0, 1, self.dim) / (np.abs(np.random.normal(0, 1, self.dim)) ** (1 / 3))
        local_best = np.clip(trial + step_size, self.bounds[0], self.bounds[1])
        local_best_fitness = func(local_best)
        self.evaluations += 1
        if local_best_fitness < func(trial):
            return local_best
        return trial
    
    def _merge_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            selected_idx = np.random.choice(sorted_indices[:half])
            populations[sorted_indices[i]] = populations[selected_idx]
            fitness[sorted_indices[i]] = fitness[selected_idx]