import numpy as np
from sklearn.cluster import KMeans

class EHDEAP_MPD_Enhanced_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_subpopulations = 5
        self.subpop_size = (12 * dim) // self.initial_subpopulations
        self.bounds = (-5.0, 5.0)
        self.base_mutation_factor = 0.5
        self.crossover_prob = 0.8
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
                self._cluster_and_update(population, fitness[s], func)
                
                if self.evaluations % self.merging_interval == 0:
                    self._merge_subpopulations(populations, fitness)

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

    def _adaptive_mutation_factor(self):
        return self.base_mutation_factor + (0.3 * (1 - self.evaluations / self.budget))

    def _elite_mutate(self, a, elite, b, c):
        mutation_factor = self._adaptive_mutation_factor()
        mutant = np.clip(a + mutation_factor * (elite - b + b - c), self.bounds[0], self.bounds[1])
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

    def _merge_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            selected_idx = np.random.choice(sorted_indices[:half])
            populations[sorted_indices[i]] = populations[selected_idx]
            fitness[sorted_indices[i]] = fitness[selected_idx]

    def _cluster_and_update(self, population, fitness, func):
        kmeans = KMeans(n_clusters=min(5, len(population)//2))
        kmeans.fit(population)
        cluster_labels = kmeans.labels_
        for i in range(self.subpop_size):
            if self.evaluations >= self.budget:
                break
            cluster_indices = np.where(cluster_labels == cluster_labels[i])[0]
            elite_indices = np.argsort(fitness[cluster_indices])[:int(self.elite_fraction * len(cluster_indices))]
            elite = population[np.random.choice(elite_indices)]
            indices = list(range(self.subpop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self._elite_mutate(population[a], elite, population[b], population[c])

            trial = self._crossover(population[i], mutant)
            refined_trial = self._levy_local_search(trial, func)

            trial_fitness = func(refined_trial)
            self.evaluations += 1

            if trial_fitness < fitness[i]:
                population[i] = refined_trial
                fitness[i] = trial_fitness