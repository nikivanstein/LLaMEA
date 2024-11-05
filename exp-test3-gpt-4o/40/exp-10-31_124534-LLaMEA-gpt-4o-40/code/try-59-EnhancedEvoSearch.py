import numpy as np

class EnhancedEvoSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_subpopulations = 4
        self.subpop_size = (10 * dim) // self.initial_subpopulations
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.9
        self.base_crossover_prob = 0.85
        self.crossover_increment = 0.05
        self.evaluations = 0
        self.top_elite_fraction = 0.2
        self.secondary_elite_fraction = 0.4
        self.adaptive_interval = 50

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

                    top_elite_indices = np.argsort(fitness[s])[:int(self.top_elite_fraction * self.subpop_size)]
                    secondary_elite_indices = np.argsort(fitness[s])[int(self.top_elite_fraction * self.subpop_size):int(self.secondary_elite_fraction * self.subpop_size)]
                    elite = population[np.random.choice(top_elite_indices)]
                    secondary_elite = population[np.random.choice(secondary_elite_indices)]
                    indices = list(range(self.subpop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = self._adaptive_mutate(population[a], elite, secondary_elite, population[b], population[c])

                    self.crossover_prob = min(1.0, self.base_crossover_prob + (self.crossover_increment * (self.evaluations / self.budget)))
                    trial = self._crossover(population[i], mutant)

                    trial = self._dynamic_local_search(trial, func)

                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = trial
                        fitness[s][i] = trial_fitness

                if self.evaluations % self.adaptive_interval == 0:
                    self._balance_subpopulations(populations, fitness)

            if subpopulations > 1 and self.evaluations % (5 * self.adaptive_interval) == 0:
                subpopulations = max(1, subpopulations // 2)
                populations = populations[:subpopulations]
                fitness = fitness[:subpopulations]

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _adaptive_mutate(self, a, elite, secondary_elite, b, c):
        mutant = np.clip(a + self.mutation_factor * (elite - b + secondary_elite - c), self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _dynamic_local_search(self, trial, func):
        gamma = 0.01
        step_size = gamma * np.random.normal(0, 1, self.dim) / (np.abs(np.random.normal(0, 1, self.dim)) ** (1 / 5))
        local_best = np.clip(trial + step_size, self.bounds[0], self.bounds[1])
        local_best_fitness = func(local_best)
        self.evaluations += 1
        if local_best_fitness < func(trial):
            return local_best
        return trial
    
    def _balance_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            selected_idx = np.random.choice(sorted_indices[:half])
            populations[sorted_indices[i]] = populations[selected_idx]
            fitness[sorted_indices[i]] = fitness[selected_idx]