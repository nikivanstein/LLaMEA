import numpy as np

class QuantumInspiredAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // (3 * dim))
        self.mutation_factor = 0.7
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def quantum_superposition(self):
        return self.mutation_factor * np.random.randn(self.dim)

    def adaptive_population_control(self):
        return max(10, int(self.population_size * (0.75 + 0.5 * np.random.rand())))

    def quantum_differential_evolution(self, func):
        self.population_size = self.adaptive_population_control()
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_vector = np.clip(a + self.quantum_superposition() * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutation_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def adaptive_hill_climbing(self, func):
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

    def __call__(self, func):
        self.quantum_differential_evolution(func)
        self.adaptive_hill_climbing(func)
        return self.best_solution