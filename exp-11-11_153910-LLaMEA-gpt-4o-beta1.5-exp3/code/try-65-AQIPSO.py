import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_velocity_and_position(self):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        
        cognitive_component = self.c1 * r1 * (self.personal_best_position - self.position)
        social_component = self.c2 * r2 * (self.global_best_position - self.position)
        
        self.velocity = self.w * self.velocity + cognitive_component + social_component
        self.position += self.velocity

        # Quantum-inspired update
        delta = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        q = np.minimum(1, np.exp(-np.linalg.norm(self.velocity, axis=1)))
        quantum_jump = q[:, np.newaxis] * delta
        self.position += quantum_jump

        self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.position[i])
            self.personal_best_fitness[i] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.position[i]

        while self.evaluations < self.budget:
            self.update_velocity_and_position()

            for i in range(self.pop_size):
                fitness = self.evaluate(func, self.position[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_position[i] = self.position[i]
                    
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.position[i]

        return self.global_best_position