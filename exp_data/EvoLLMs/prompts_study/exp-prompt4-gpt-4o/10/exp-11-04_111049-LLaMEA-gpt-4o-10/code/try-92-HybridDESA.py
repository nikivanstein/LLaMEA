import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1200.0  # Adjusted initial temperature
        self.cooling_rate = 0.92  # Adjusted cooling rate

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound,
                                 (self.population_size, self.dim))

    def _adaptive_mutation_factor(self, evaluations):
        return 0.5 + 0.5 * (1 - evaluations / self.budget)

    def _chaotic_initialization(self, size):
        x = np.random.rand(size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * np.abs(np.sin(np.pi * x**2))

    def _mutate(self, idx, population, evaluations):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_factor = self._adaptive_mutation_factor(evaluations)
        mutant = np.clip(population[a] + adaptive_factor * (population[b] - population[c]),
                         self.lower_bound, self.upper_bound)
        return mutant

    def _adaptive_crossover_rate(self, evaluations):
        return 0.6 + 0.35 * (evaluations / self.budget)

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
        return max(4, int(self.initial_population_size * (0.4 + 0.6 * np.cos(np.pi * evaluations / self.budget))))  # Enhanced dynamic adaptation

    def _stochastic_ranking(self, fitness):
        indices = np.argsort(fitness)
        return indices

    def _crowding_distance(self, population):
        distances = np.zeros(self.population_size)
        for i in range(self.dim):
            sorted_idx = np.argsort(population[:, i])
            sorted_population = population[sorted_idx, i]
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')
            for j in range(1, self.population_size - 1):
                distances[sorted_idx[j]] += (sorted_population[j + 1] - sorted_population[j - 1])
        return distances

    def __call__(self, func):
        population = self._chaotic_initialization(self.population_size)
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        local_search_frequency = max(1, int(0.1 * self.budget / self.initial_population_size))
        
        while evaluations < self.budget:
            self.population_size = self._adaptive_population_size(evaluations)
            
            if evaluations % local_search_frequency == 0:
                local_search_idx = np.random.randint(self.population_size)
                local_search_candidate = np.clip(population[local_search_idx] + 
                                                 np.random.normal(0, 0.5, self.dim),  # Enhanced local search modification
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
            crowd_indices = np.argsort(self._crowding_distance(population))
            combined_indices = np.lexsort((crowd_indices, ranked_indices))
            population = np.vstack((population[combined_indices[:self.population_size]], best_solution))  # Add best solution for elitism
            fitness = np.append(fitness[combined_indices[:self.population_size]], best_fitness)  # Maintain corresponding fitness

        return best_solution