import numpy as np

class QuantumInspiredAdaptiveEvolutionarySwarmOptimization:
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
        self.f = 0.8  # Slightly increased mutation factor for exploration
        self.cr = 0.85  # Modulated crossover probability for balance
        self.w = 0.6  # Increased inertia weight for exploration
        self.c1 = 1.3  # Adjusted cognitive coefficient
        self.c2 = 1.7  # Enhanced social coefficient for improved global search

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Quantum-inspired State Update
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant_vector = x0 + self.f * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, self.population[i])

                # Selection
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

            # Enhanced Particle Swarm Optimization with Quantum-inspired Parameters
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                phi = np.pi * np.random.rand(self.dim)  # Quantum phase
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * np.sin(phi) * (self.personal_best_positions[i] - self.population[i])
                                      + self.c2 * r2 * np.cos(phi) * (self.global_best_position - self.population[i]))
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate
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