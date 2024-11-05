import numpy as np

class EHDEAP_MPD_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_subpopulations = 5
        self.subpop_size = (12 * dim) // self.initial_subpopulations
        self.bounds = (-5.0, 5.0)
        self.base_mutation_factor = 0.5  # Adjusted mutation factor
        self.crossover_prob = 0.9
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
                    current_mutation_factor = self._adaptive_mutation_factor(fitness[s][i], np.mean(fitness[s]))

                    mutant = self._mutate(population[a], elite, population[b], population[c], current_mutation_factor)
                    
                    trial = self._crossover(population[i], mutant)
                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[s][i]:
                        population[i] = trial
                        fitness[s][i] = trial_fitness

                    trial_fitness = min(trial_fitness, self._multi_trial_search(trial, func, fitness[s][i]))

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

    def _mutate(self, a, elite, b, c, mutation_factor):
        mutant = np.clip(a + mutation_factor * (elite - b + b - c), self.bounds[0], self.bounds[1])
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _adaptive_mutation_factor(self, current_fitness, mean_fitness):
        return self.base_mutation_factor + 0.5 * (1 - np.exp(-np.abs(current_fitness - mean_fitness)))

    def _multi_trial_search(self, trial, func, current_fitness):
        best_fitness = current_fitness
        for _ in range(3):
            perturbed_trial = self._perturb(trial)
            fitness = func(perturbed_trial)
            self.evaluations += 1
            if fitness < best_fitness:
                best_fitness = fitness
        return best_fitness

    def _perturb(self, solution):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        return np.clip(solution + perturbation, self.bounds[0], self.bounds[1])

    def _merge_subpopulations(self, populations, fitness):
        subpop_fitness = np.array([f.mean() for f in fitness])
        sorted_indices = np.argsort(subpop_fitness)
        half = len(sorted_indices) // 2
        for i in range(half, len(sorted_indices)):
            selected_idx = np.random.choice(sorted_indices[:half])
            populations[sorted_indices[i]] = populations[selected_idx]
            fitness[sorted_indices[i]] = fitness[selected_idx]