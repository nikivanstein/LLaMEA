import numpy as np

class DynamicAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Reduced population size for quicker exploration
        self.mutation_factor = 0.8  # Adjusted mutation factor for better diversity
        self.crossover_rate = 0.8  # Adjusted crossover rate for balanced exploration-exploitation
        self.local_search_intensity = 4  # Slightly reduced intensity for quicker convergence
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

    def stochastic_local_search(self, func):
        best_index = np.argmin(self.fitness)
        best_solution = self.population[best_index].copy()
        step_size = 0.15  # Increased step size for more aggressive local search
        for _ in range(self.local_search_intensity + np.random.randint(0, 2)):  # Less variance in intensity
            if self.used_budget >= self.budget:
                break
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(best_solution + perturbation, -5, 5)
            candidate_fitness = func(candidate)
            self.used_budget += 1
            if candidate_fitness < self.fitness[best_index]:
                self.fitness[best_index] = candidate_fitness
                best_solution = candidate

    def adapt_parameters(self):
        if self.used_budget > self.budget // 3:  # Earlier adaptation trigger point
            self.local_search_intensity = 8  # Increased adaptive search intensity for focused improvement
            self.mutation_factor = 0.9  # Further adaptive mutation factor for exploration
            self.crossover_rate = 0.95  # Enhanced crossover rate for exploiting good solutions

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.differential_evolution(func)
            self.stochastic_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]