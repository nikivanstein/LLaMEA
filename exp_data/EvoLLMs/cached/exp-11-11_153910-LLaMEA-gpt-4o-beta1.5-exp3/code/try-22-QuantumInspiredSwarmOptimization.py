import numpy as np

class QuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.local_best_positions = np.copy(self.population)
        self.local_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_particle(self, i, func):
        rand_cognitive = np.random.rand(self.dim)
        rand_social = np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coefficient * rand_cognitive * (self.local_best_positions[i] - self.population[i])
        social_velocity = self.social_coefficient * rand_social * (self.global_best_position - self.population[i])
        self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
        self.population[i] += self.velocities[i]
        self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        fitness = self.evaluate(func, self.population[i])
        if fitness < self.local_best_fitness[i]:
            self.local_best_fitness[i] = fitness
            self.local_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

    def quantum_superposition(self):
        superposition_prob = np.random.rand(self.pop_size, self.dim)
        for i in range(self.pop_size):
            if np.random.rand() < 0.1:  # 10% chance to explore quantum state
                self.population[i] = superposition_prob[i] * self.lower_bound + (1 - superposition_prob[i]) * self.upper_bound
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.local_best_fitness[i] = fitness
            self.local_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            self.quantum_superposition()
            for i in range(self.pop_size):
                self.update_particle(i, func)

        return self.global_best_position