import numpy as np

class MutationAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Adjusted initial population size
        self.mutation_factor = 0.7
        self.crossover_rate = 0.85
        self.local_search_intensity = 5
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
        step_size = 0.15  # Increased step size for exploration
        for _ in range(self.local_search_intensity + np.random.randint(0, 3)):
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
        if self.used_budget > self.budget // 3:  # Earlier adaptation of parameters
            self.local_search_intensity = 10
            self.mutation_factor = 0.85
            self.crossover_rate = 0.9
            if self.used_budget > 2 * self.budget // 3:
                self.population_size = int(self.population_size * 0.75)  # Dynamic reduction of population size

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.used_budget = self.population_size
        while self.used_budget < self.budget:
            self.differential_evolution(func)
            self.stochastic_local_search(func)
            self.adapt_parameters()
        best_index = np.argmin(self.fitness)
        return self.population[best_index]