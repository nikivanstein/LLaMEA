import numpy as np
import concurrent.futures

class DynamicBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.frequency_min = 0
        self.frequency_max = 2
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')

    def __call__(self, func):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
                futures = [executor.submit(self.evaluate_solution, func, idx, frequencies[idx]) for idx in range(self.population_size)]
                for future in concurrent.futures.as_completed(futures):
                    idx, fitness = future.result()
                    if fitness < self.best_fitness:
                        self.best_solution = self.population[idx]
                        self.best_fitness = fitness
        return self.best_solution

    def evaluate_solution(self, func, idx, frequency):
        if np.random.rand() > self.pulse_rate:
            self.velocities[idx] += (self.population[idx] - self.best_solution) * frequency
        self.population[idx] += self.velocities[idx]
        fitness = func(self.population[idx])
        return idx, fitness