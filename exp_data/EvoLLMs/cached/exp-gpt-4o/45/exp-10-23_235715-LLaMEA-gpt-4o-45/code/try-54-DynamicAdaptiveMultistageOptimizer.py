import numpy as np

class DynamicAdaptiveMultistageOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(60, self.budget // (3 * dim))
        self.mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def dynamic_mutation_factor(self, diversity):
        return 0.2 + 0.6 * np.exp(-diversity)

    def calculate_diversity(self):
        return np.mean(np.std(self.population, axis=0))

    def adaptive_differential_evolution(self, func):
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            diversity = self.calculate_diversity()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_factor = self.dynamic_mutation_factor(diversity)
                mutant_vector = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def adaptive_stochastic_hill_climbing(self, func):
        step_size = 0.1
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                direction = np.random.uniform(-1, 1, self.dim)
                candidate = self.population[i] + step_size * direction
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate
                    step_size *= 1.05
                else:
                    step_size *= 0.95

    def __call__(self, func):
        self.adaptive_differential_evolution(func)
        self.adaptive_stochastic_hill_climbing(func)
        return self.best_solution