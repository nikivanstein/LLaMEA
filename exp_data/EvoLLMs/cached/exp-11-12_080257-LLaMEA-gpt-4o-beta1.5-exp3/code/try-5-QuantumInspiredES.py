import numpy as np

class QuantumInspiredES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.quantum_register = np.pi * np.random.rand(self.population_size, self.dim)
        self.sigma = 0.1
        self.learning_rate = 0.9

    def decode(self):
        # Quantum superposition to real values
        return self.lower_bound + (self.upper_bound - self.lower_bound) * (np.sin(self.quantum_register) ** 2)

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def update_quantum_register(self, population, best, best_fitness):
        for i in range(self.population_size):
            delta = self.learning_rate * (population[i] - best)
            delta = np.clip(delta, -self.sigma, self.sigma)
            self.quantum_register[i] += delta * (1 if np.random.rand() > 0.5 else -1)
            self.quantum_register[i] = np.mod(self.quantum_register[i], np.pi)

    def __call__(self, func):
        best = None
        best_fitness = float('inf')

        while self.eval_count < self.budget:
            population = self.decode()
            fitness = self.evaluate_population(population, func)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best = population[current_best_idx]

            self.update_quantum_register(population, best, best_fitness)

            if self.eval_count + self.population_size > self.budget:
                break

        return best