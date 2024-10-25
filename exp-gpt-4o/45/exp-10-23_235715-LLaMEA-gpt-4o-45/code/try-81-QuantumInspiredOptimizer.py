import numpy as np

class QuantumInspiredOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // (2 * dim))
        self.mutation_factor = 0.7
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def dynamic_mutation_factor(self):
        return 0.4 + 0.4 * np.random.rand()

    def quantum_superposition(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def differential_evolution(self, func):
        for _ in range(self.budget // (2 * self.population_size)):
            if self.evaluations >= self.budget:
                break
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[idxs]
                mutation_factor = self.dynamic_mutation_factor()
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

    def quantum_entanglement(self, func):
        for _ in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            entangled_state = self.quantum_superposition()
            entangled_fitness = func(entangled_state)
            self.evaluations += 1
            if entangled_fitness < self.best_fitness:
                self.best_fitness = entangled_fitness
                self.best_solution = entangled_state

    def __call__(self, func):
        self.differential_evolution(func)
        self.quantum_entanglement(func)
        return self.best_solution