import numpy as np

class HybridQuantumInspiredEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 25  # Slightly increased population for diversified search
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Fine-tuned velocity range
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.eval_count = 0
        self.f = 0.8  # Increased mutation factor for extensive exploration
        self.cr = 0.85  # Lower crossover probability for strategic diversity
        self.w = 0.3  # Further reduced inertia for faster convergence
        self.c1 = 1.7  # Enhanced cognitive coefficient to boost personal achievement
        self.c2 = 1.5  # Adjusted social coefficient for effective global search

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Adaptive Differential Evolution with Quantum-inspired Mutation
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[indices]
                mutant_vector = x0 + self.f * (x1 - x2)
                q_bit_flip = np.random.rand(self.dim) < 0.1  # Quantum-inspired bit flip
                mutant_vector = np.where(q_bit_flip, -mutant_vector, mutant_vector)
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

            # Enhanced Particle Swarm Optimization with Quantum Jump
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                                      + self.c2 * r2 * (self.global_best_position - self.population[i]))
                quantum_jump = np.random.rand(self.dim) < 0.05  # Quantum jump for escaping local optima
                self.population[i] += np.where(quantum_jump, np.random.uniform(self.lower_bound, self.upper_bound, self.dim), self.velocities[i])
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