import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased population size for diverse exploration
        self.F_min = 0.4  # Minimum scaling factor for mutation
        self.F_max = 0.9  # Maximum scaling factor for mutation
        self.CR = 0.85  # Lower crossover probability
        self.local_search_prob_init = 0.2  # Initial local search probability

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx, evals):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        F_dynamic = self.F_min + (self.F_max - self.F_min) * evals / self.budget
        mutant = np.clip(a + F_dynamic * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.05, 0.15)
        new_individual = np.clip(individual + step_size * direction, self.lower_bound, self.upper_bound)
        return new_individual

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(population, i, evals)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

                local_search_prob_dynamic = self.local_search_prob_init + 0.6 * (evals / self.budget)
                if np.random.rand() < local_search_prob_dynamic:
                    local_candidate = self._local_search(population[i])
                    local_fitness = func(local_candidate)
                    evals += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

                if evals >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]