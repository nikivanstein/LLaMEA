import numpy as np

class HDEL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self._mutate(population[a], population[b], population[c])

                # Crossover
                trial = self._crossover(population[i], mutant)

                # Local Search
                refined_trial = self._local_search(trial, func)

                # Selection
                trial_fitness = func(refined_trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = refined_trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

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
        step_size = 0.01
        local_best = trial
        local_best_fitness = func(local_best)
        for _ in range(5):  # limited steps for local search
            candidate = local_best + step_size * np.random.normal(0, 1, self.dim)
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < local_best_fitness:
                local_best = candidate
                local_best_fitness = candidate_fitness
        return local_best