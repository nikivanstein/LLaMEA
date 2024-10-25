import numpy as np

class QuantumInspiredAdaptiveMultistageOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // (3 * dim))
        self.mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def quantum_superposition(self):
        return np.random.rand(self.dim) * 2 - 1

    def quantum_entanglement(self, a, b):
        return a * 0.5 + b * 0.5

    def differential_evolution(self, func):
        for _ in range(self.budget // (3 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial_vector

    def stochastic_hill_climbing(self, func):
        step_size = 0.1
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                direction = self.quantum_superposition()
                candidate = self.population[i] + step_size * direction
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate
                if candidate_fitness < func(self.population[i]):
                    self.population[i] = candidate

    def quantum_interference(self, func):
        for _ in range(self.population_size // 2):
            if self.evaluations >= self.budget:
                break
            idxs = np.random.choice(self.population_size, 2, replace=False)
            a, b = self.population[idxs]
            entangled_state = self.quantum_entanglement(a, b)
            candidate_fitness = func(entangled_state)
            self.evaluations += 1
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = entangled_state

    def __call__(self, func):
        self.differential_evolution(func)
        self.stochastic_hill_climbing(func)
        self.quantum_interference(func)
        return self.best_solution