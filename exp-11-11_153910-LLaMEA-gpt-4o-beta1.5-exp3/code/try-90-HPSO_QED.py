import numpy as np

class HPSO_QED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_local_positions = np.copy(self.population)
        self.best_local_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.q_crossover_rate = 0.2  # Quantum-inspired crossover rate

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_crossover(self, x1, x2):
        return 0.5 * (x1 + x2) + np.random.normal(scale=self.q_crossover_rate, size=x1.shape)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_local_positions[i] = self.population[i]
            self.best_local_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.velocities[i]
                                      + self.c1 * r1 * (self.best_local_positions[i] - self.population[i])
                                      + self.c2 * r2 * (self.best_global_position - self.population[i]))
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                candidate = self.quantum_crossover(self.population[i], self.best_global_position)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                
                candidate_fitness = self.evaluate(func, candidate)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = candidate_fitness
                    if candidate_fitness < self.best_local_fitness[i]:
                        self.best_local_fitness[i] = candidate_fitness
                        self.best_local_positions[i] = candidate
                    if candidate_fitness < self.best_global_fitness:
                        self.best_global_fitness = candidate_fitness
                        self.best_global_position = candidate

        return self.best_global_position