import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.alpha = 0.99  # decay factor for updating quantum bits
        self.beta = 0.9    # probability threshold for quantum-inspired crossover

    def __call__(self, func):
        q_population = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        population = self.quantum_measure(q_population)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.pop_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.permutation(self.pop_size)
                p1, p2 = q_population[indices[:2]]

                quantum_crossover = np.random.rand(self.dim) < self.beta
                q_offspring = np.where(quantum_crossover, self.alpha * p1 + (1 - self.alpha) * p2, q_population[i])
                offspring = self.quantum_measure(q_offspring)

                trial_fitness = func(offspring)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    q_population[i] = q_offspring
                    population[i] = offspring
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = trial_fitness

        return best_individual, best_fitness

    def quantum_measure(self, q_bits):
        amplitudes = np.abs(q_bits) / np.sqrt(np.sum(q_bits**2, axis=1, keepdims=True))
        measured = np.sign(q_bits) * (self.lower_bound + (amplitudes * (self.upper_bound - self.lower_bound)))
        return np.clip(measured, self.lower_bound, self.upper_bound)