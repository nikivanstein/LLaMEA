import numpy as np

class QuantumInspiredEvolutionarySwarmOptimizationAdaptiveParameters:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.eval_count = 0
        self.f = 0.75
        self.cr = 0.9
        self.w = 0.7  # Adjusted for better exploration
        self.c1 = 1.5
        self.c2 = 1.4

    def quantum_superposition(self, x0, x1, x2):
        alpha = np.random.rand()
        return alpha * x0 + (1 - alpha) * (x1 - x2)

    def update_parameters(self):
        self.w = 0.4 + 0.3 * (1 - self.eval_count / self.budget)  # Linearly decreasing inertia weight
        self.c1 = 1.5 + 0.5 * (self.eval_count / self.budget)  # Increasing cognitive component
        self.c2 = 1.4 - 0.4 * (self.eval_count / self.budget)  # Decreasing social component

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Quantum-inspired Differential Evolution
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant_vector = self.quantum_superposition(x0, x1, x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, self.population[i])

                trial_score = func(trial_vector)
                self.eval_count += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

                if self.eval_count >= self.budget:
                    break

            # Adaptive Parameter Update
            self.update_parameters()

            # Enhanced Particle Swarm Optimization
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                                      + self.c2 * r2 * (self.global_best_position - self.population[i]))
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                score = func(self.population[i])
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

                if self.eval_count >= self.budget:
                    break

        return self.global_best_position, self.global_best_score