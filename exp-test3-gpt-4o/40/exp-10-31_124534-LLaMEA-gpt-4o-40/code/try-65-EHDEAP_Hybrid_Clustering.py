import numpy as np

class EHDEAP_Hybrid_Clustering:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_subpopulations = 4  # Reduced initial subpopulations
        self.subpop_size = (10 * dim) // self.initial_subpopulations  # Adjusted subpopulation size
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.85  # Adjusted mutation factor
        self.base_crossover_prob = 0.85  # Adjusted crossover probability
        self.crossover_decay_rate = 0.05  # Adjusted decay rate
        self.evaluations = 0
        self.elite_fraction = 0.3
        self.merging_interval = 80  # Reduced merging interval
        self.constriction_factor = 0.729  # Constriction factor for convergence

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
                    mutant = self._constriction_mutate(population[a], elite, population[b], population[c])

                    self.crossover_prob = max(0.1, self.base_crossover_prob - (self.crossover_decay_rate * (self.evaluations / self.budget)))
                    trial = self._crossover(population[i], mutant)

                    refined_trial = self._adaptive_clustering_search(trial, func)

                    trial_fitness = func(refined_trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = refined_trial
                        fitness[s][i] = trial_fitness

                if self.evaluations % self.merging_interval == 0:
                    self._merge_subpopulations(populations, fitness)

            if subpopulations > 1 and self.evaluations % (4 * self.merging_interval) == 0:
                subpopulations = max(1, subpopulations // 2)
                populations = populations[:subpopulations]
                fitness = fitness[:subpopulations]

        best_idx = np.argmin([f.min() for f in fitness])
        best_subpop = populations[best_idx]
        best_fit_idx = np.argmin(fitness[best_idx])
        return best_subpop[best_fit_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.subpop_size, self.dim))

    def _constriction_mutate(self, a, elite, b, c):
        mutant = np.clip(a + self.constriction_factor * self.mutation_factor * (elite - b + b - c), self.bounds[0], self.bounds[1])
        return mutant

    def _adaptive_clustering_search(self, trial, func):
        cluster_radius = 0.1
        perturbation = cluster_radius * np.random.normal(0, 1, self.dim)
        local_best = np.clip(trial + perturbation, self.bounds[0], self.bounds[1])
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