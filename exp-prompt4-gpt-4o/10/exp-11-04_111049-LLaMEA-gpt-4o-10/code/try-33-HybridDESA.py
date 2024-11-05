import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim  # Common heuristic for DE
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1000.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.97

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound,
                                 (self.population_size, self.dim))

    def _adaptive_mutation_factor(self, evaluations):
        return 0.5 + 0.3 * (1 - evaluations / self.budget)

    def _mutate(self, idx, population, evaluations):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_factor = self._adaptive_mutation_factor(evaluations)
        mutant = np.clip(population[a] + adaptive_factor * (population[b] - population[c]),
                         self.lower_bound, self.upper_bound)
        return mutant

    def _adaptive_crossover_rate(self, evaluations):
        return 0.6 + 0.3 * (evaluations / self.budget)

    def _crossover(self, target, mutant, evaluations):
        crossover_rate = self._adaptive_crossover_rate(evaluations)
        crossover_mask = np.random.rand(self.dim) < crossover_rate
        crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)

    def _adaptive_population_size(self, evaluations):
        return max(4, int(self.initial_population_size * (1 - evaluations / self.budget)**0.5))

    def _stochastic_ranking(self, fitness):
        indices = np.argsort(fitness)
        return indices

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            self.population_size = self._adaptive_population_size(evaluations)
            
            # Local search enhancement
            local_search_idx = np.random.randint(self.population_size)
            local_search_candidate = np.clip(population[local_search_idx] + 
                                             np.random.normal(0, 0.1, self.dim), 
                                             self.lower_bound, self.upper_bound)
            local_search_fitness = func(local_search_candidate)
            evaluations += 1
            if local_search_fitness < fitness[local_search_idx]:
                population[local_search_idx] = local_search_candidate
                fitness[local_search_idx] = local_search_fitness

            for i in range(self.population_size):
                mutant = self._mutate(i, population, evaluations)
                trial = self._crossover(population[i], mutant, evaluations)
                trial_fitness = func(trial)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if trial_fitness < fitness[i] or np.random.rand() < self._acceptance_probability(fitness[i], trial_fitness, self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness

            self.temperature *= self.cooling_rate
            ranked_indices = self._stochastic_ranking(fitness)
            population = population[ranked_indices[:self.population_size]]
            fitness = fitness[ranked_indices[:self.population_size]]

        return best_solution