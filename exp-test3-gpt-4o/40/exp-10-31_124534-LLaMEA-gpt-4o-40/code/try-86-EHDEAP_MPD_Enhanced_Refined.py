import numpy as np

class EHDEAP_MPD_Enhanced_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_subpopulations = 5
        self.subpop_size = (12 * dim) // self.initial_subpopulations
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.mutation_decay_rate = 0.2 # New adaptive mutation rate
        self.base_crossover_prob = 0.9
        self.crossover_decay_rate = 0.1
        self.evaluations = 0
        self.elite_fraction = 0.3
        self.merging_interval = 100

    def __call__(self, func):
        subpopulations = self.initial_subpopulations
        populations = [self._initialize_population() for _ in range(subpopulations)]
        fitness = [np.array([func(ind) for ind in pop]) for pop in populations]
        self.evaluations += self.subpop_size * subpopulations

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
                    self.mutation_factor = max(0.4, self.mutation_factor - (self.mutation_decay_rate * (self.evaluations / self.budget))) # Adaptive mutation
                    mutant = self._elite_mutate(population[a], elite, population[b], population[c])

                    self.crossover_prob = max(0.1, self.base_crossover_prob - (self.crossover_decay_rate * (self.evaluations / self.budget)))
                    trial = self._crossover(population[i], mutant)

                    refined_trial = self._levy_local_search(trial, func)

                    trial_fitness = func(refined_trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = refined_trial
                        fitness[s][i] = trial_fitness

                if self.evaluations % self.merging_interval == 0:
                    self._dynamic_merge_subpopulations(populations, fitness) # Improved merging strategy

            if subpopulations > 1 and self.evaluations % (5 * self.merging_interval) == 0:
                subpopulations = max(1, subpopulations // 2)
                populations = populations[:subpopulations]
                fitness = fitness[:subpopulations]

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _elite_mutate(self, a, elite, b, c):
        mutant = np.clip(a + self.mutation_factor * (elite - b + b - c), self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _levy_local_search(self, trial, func):
        alpha = 0.02
        step_size = alpha * np.random.normal(0, 1, self.dim) / (np.abs(np.random.normal(0, 1, self.dim)) ** (1 / 3))
        local_best = np.clip(trial + step_size, self.bounds[0], self.bounds[1])
        local_best_fitness = func(local_best)
        self.evaluations += 1
        if local_best_fitness < func(trial):
            return local_best
        return trial
    
    def _dynamic_merge_subpopulations(self, populations, fitness): # Improved merging strategy
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        worst_idx = sorted_indices[-1]
        best_idx = sorted_indices[0]
        populations[worst_idx] = populations[best_idx] + np.random.normal(0, 0.1, populations[best_idx].shape)
        fitness[worst_idx] = np.array([func(ind) for ind in populations[worst_idx]])