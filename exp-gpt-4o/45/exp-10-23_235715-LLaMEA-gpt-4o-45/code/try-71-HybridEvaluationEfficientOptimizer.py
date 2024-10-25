import numpy as np

class HybridEvaluationEfficientOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // (2 * dim))
        self.mutation_factor_range = (0.4, 0.9)
        self.crossover_rate = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0
        self.learning_rate = 0.1

    def adaptive_mutation_factor(self):
        return self.mutation_factor_range[0] + (self.mutation_factor_range[1] - self.mutation_factor_range[0]) * np.random.rand()

    def enhanced_differential_evolution(self, func):
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_factor = self.adaptive_mutation_factor()
                mutant_vector = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                    self.learning_rate = max(0.01, self.learning_rate * 0.95)
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def adaptive_hill_climbing(self, func):
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                direction = np.random.uniform(-1, 1, self.dim)
                candidate = self.population[i] + self.learning_rate * direction
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                    self.learning_rate = max(0.01, self.learning_rate * 0.95)
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate

    def __call__(self, func):
        self.enhanced_differential_evolution(func)
        self.adaptive_hill_climbing(func)
        return self.best_solution