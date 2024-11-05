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
        self.temperature = 1000.0
        self.cooling_rate = 0.95

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound,
                                 (self.population_size, self.dim))

    def _adaptive_mutation_factor(self, evaluations):
        return 0.5 + 0.5 * (1 - evaluations / self.budget)

    def _chaotic_initialization(self, size):
        x = np.random.rand(size, self.dim)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * np.abs(np.sin(np.pi * x**2))

    def _levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.1 * step  # Levy flight multiplier

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
        return max(4, int(self.initial_population_size * (0.5 + 0.5 * np.cos(np.pi * evaluations / self.budget))))  # Modified dynamic adaptation

    def _stochastic_ranking(self, fitness):
        indices = np.argsort(fitness)
        return indices

    def __call__(self, func):
        population = self._chaotic_initialization(self.population_size)
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            self.population_size = self._adaptive_population_size(evaluations)
            
            local_search_idx = np.random.randint(self.population_size)
            local_search_candidate = np.clip(population[local_search_idx] + 
                                             np.random.normal(0, 0.25, self.dim) + self._levy_flight(self.dim),  # Added Lévy flight
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
            population = np.vstack((population[ranked_indices[:self.population_size]], best_solution))  # Add best solution for elitism
            fitness = np.append(fitness[ranked_indices[:self.population_size]], best_fitness)  # Maintain corresponding fitness

        return best_solution