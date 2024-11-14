import numpy as np

class AdaptiveQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.individuals = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_individual = np.copy(self.individuals[0])
        self.best_score = float('inf')
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate current population
            scores = np.array([func(ind) for ind in self.individuals])
            self.func_evaluations += self.population_size

            # Update global best
            min_index = np.argmin(scores)
            if scores[min_index] < self.best_score:
                self.best_individual = self.individuals[min_index]
                self.best_score = scores[min_index]

            # Quantum-inspired rotation gate
            theta = np.random.uniform(0, np.pi, (self.population_size, self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            
            for i in range(self.population_size):
                probability_amplitude = np.random.random(self.dim)
                qubit = np.array([np.sqrt(probability_amplitude), np.sqrt(1 - probability_amplitude)])
                rotated_qubit = rotation_matrix[i] @ qubit
                self.individuals[i] = self.lower_bound + (self.upper_bound - self.lower_bound) * rotated_qubit[0]

            # Adaptive crossover based on diversity
            diversity = np.std(self.individuals, axis=0)
            crossover_rate = np.clip(0.2 + 0.8 * (1 - diversity / np.ptp(self.individuals, axis=0)), 0.2, 0.9)

            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size and np.random.random() < crossover_rate.mean():
                    crossover_point = np.random.randint(1, self.dim)
                    self.individuals[i][:crossover_point], self.individuals[i+1][:crossover_point] = (
                        self.individuals[i+1][:crossover_point].copy(),
                        self.individuals[i][:crossover_point].copy(),
                    )

            # Boundary check
            self.individuals = np.clip(self.individuals, self.lower_bound, self.upper_bound)

        return self.best_individual