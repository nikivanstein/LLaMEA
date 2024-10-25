import numpy as np

class MultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Adjusted population size
        self.mutation_factor = 0.65  # Adjusted mutation factor
        self.crossover_rate = 0.80  # Adjusted crossover rate
        self.local_search_intensity = 7  # Modified intensity
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0

    def differential_evolution(self, func):
        for i in range(self.population_size):
            if self.used_budget >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.mutation_factor * (b - c), -5, 5)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            self.used_budget += 1
            if trial_fitness < self.fitness[i]:
                self.fitness[i] = trial_fitness
                self.population[i] = trial_vector

    def adaptive_local_search(self, func):
        best_indices = np.argsort(self.fitness)[:self.local_search_intensity]
        for idx in best_indices:
            if self.used_budget >= self.budget:
                break
            step_size = np.clip(0.1 * (1.0 - self.used_budget / self.budget), 0.01, 0.1)  # Adaptive step size
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(self.population[idx] + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[idx]:
                self.fitness[idx] = candidate_fitness
                self.population[idx] = candidate

    def adapt_parameters(self):
        if np.std(self.fitness) < 0.4:  # Adaptive check condition
            self.local_search_intensity = 9  # Adjusted search intensity
            self.mutation_factor = 0.70  # Adjusted mutation factor
            self.crossover_rate = 0.85  # Adjust crossover rate

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.differential_evolution(func)
            self.adaptive_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]