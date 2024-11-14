import numpy as np

class AdaptiveParticleSwarmOptimizationQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.7  # Inertia weight
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current position
                score = func(self.population[i])
                self.func_evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.population)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.population)
            self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity

            # Quantum-inspired perturbation
            quantum_perturbation = np.random.normal(0, 0.1, (self.population_size, self.dim))
            self.velocities += quantum_perturbation * np.exp(-0.1 * self.func_evaluations / self.budget)
            
            # Update positions
            self.population += self.velocities
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            # Adaptive inertia weight
            self.w = 0.7 - 0.4 * (self.func_evaluations / self.budget)

        return self.global_best_position