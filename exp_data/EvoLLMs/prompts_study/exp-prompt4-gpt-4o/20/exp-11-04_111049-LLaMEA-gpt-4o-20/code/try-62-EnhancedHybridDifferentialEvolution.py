import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim
        self.F = 0.5
        self.CR = 0.85
        self.local_search_prob = 0.4
        self.adaptive_alpha = 0.5
        self.num_subpopulations = 3  # New: Number of subpopulations
        self.competitive_F = [0.4, 0.6, 0.8]  # New: competitive F values

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx, F):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant1 = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
        d, e, f = pop[np.random.choice(indices, 3, replace=False)]
        mutant2 = np.clip(d + F * (e - f), self.lower_bound, self.upper_bound)
        return mutant1 if np.random.rand() > 0.5 else mutant2

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual, scale=1.0):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.05, 0.15) * scale
        new_individual = np.clip(individual + step_size * direction, self.lower_bound, self.upper_bound)
        return new_individual

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        subpop_size = self.population_size // self.num_subpopulations  # New: subpopulation size

        while evals < self.budget:
            for subpop_id in range(self.num_subpopulations):  # New: Iterate over subpopulations
                start_idx = subpop_id * subpop_size
                end_idx = start_idx + subpop_size
                subpop = population[start_idx:end_idx]
                sub_fitness = fitness[start_idx:end_idx]
                F = self.competitive_F[subpop_id]  # New: Use competitive F

                for i in range(subpop_size):
                    mutant = self._mutate(subpop, i, F)
                    trial = self._crossover(subpop[i], mutant)
                    trial_fitness = func(trial)
                    evals += 1

                    if trial_fitness < sub_fitness[i]:
                        subpop[i] = trial
                        sub_fitness[i] = trial_fitness

                    if evals >= self.budget:
                        break

                    if np.random.rand() < self.local_search_prob:
                        scale_factor = 1 - (evals / self.budget)
                        local_candidate = self._local_search(subpop[i], scale=scale_factor)
                        local_fitness = func(local_candidate)
                        evals += 1
                        if local_fitness < sub_fitness[i]:
                            subpop[i] = local_candidate
                            sub_fitness[i] = local_fitness

                    if evals >= self.budget:
                        break

                population[start_idx:end_idx] = subpop
                fitness[start_idx:end_idx] = sub_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]