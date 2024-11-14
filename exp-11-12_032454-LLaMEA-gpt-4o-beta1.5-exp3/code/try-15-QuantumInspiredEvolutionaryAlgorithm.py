import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.qbits = np.pi / 4 * np.ones((self.population_size, self.dim))  # initial Q-bit angles
        self.best_solution = None
        self.best_fitness = float('inf')
        self.observation_rate = 0.1  # rate for observing quantum states

    def __call__(self, func):
        def observe(qbits):
            # Convert quantum bits to binary and then to real values
            binary_population = (np.random.rand(*qbits.shape) < np.sin(qbits)**2).astype(int)
            real_population = self.lb + (self.ub - self.lb) * binary_population
            return real_population

        num_evaluations = 0
        while num_evaluations < self.budget:
            # Observe the quantum states
            population = observe(self.qbits)
            fitness = np.array([func(ind) for ind in population])
            num_evaluations += self.population_size

            # Update the best solution found so far
            for i in range(self.population_size):
                if fitness[i] < self.best_fitness:
                    self.best_solution = population[i]
                    self.best_fitness = fitness[i]

            # Update quantum bits based on fitness
            for i in range(self.population_size):
                if fitness[i] < self.best_fitness:
                    # Apply rotation gates - use a small angle change
                    delta_theta = np.pi / 180
                    self.qbits[i] += np.sign(np.random.rand(self.dim) - 0.5) * delta_theta
                    self.qbits[i] = np.clip(self.qbits[i], 0, np.pi / 2)

            # Occasionally, re-observe with increased variability
            if np.random.rand() < self.observation_rate:
                self.qbits = np.pi / 4 * np.ones((self.population_size, self.dim))

        return self.best_solution, self.best_fitness