import numpy as np

class AdaptiveQuantumParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7  # Adjusted for better exploration
        self.cognitive_component = 1.5  # Slight adjustment
        self.social_component = 2.0  # Slight adjustment
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.adaptive_rate = 0.2  # Modified adaptive factor for more adaptation

    def quantum_behaviour(self, size):
        return np.random.normal(0, 1, size)  # Simplified quantum behavior for exploration

    def update_population_size(self):
        shrink_factor = np.log(1 + self.evaluations / self.budget)
        new_size = max(5, int(self.initial_population_size * (1 - shrink_factor)))
        if new_size < self.population_size:
            self.positions = self.positions[:new_size]
            self.velocities = self.velocities[:new_size]
            self.personal_best_positions = self.personal_best_positions[:new_size]
            self.personal_best_scores = self.personal_best_scores[:new_size]
            self.population_size = new_size

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.update_population_size()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    return self.global_best_score

                fitness = func(self.positions[i])
                self.evaluations += 1

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            inertia_weight = self.inertia_weight * np.exp(-0.5 * self.evaluations / self.budget)

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Quantum-inspired exploration
            if np.random.rand() < self.adaptive_rate:
                for i in range(self.population_size):
                    quantum_step = self.quantum_behaviour(self.dim)
                    quantum_mutant = self.positions[i] + 0.01 * quantum_step
                    quantum_mutant = np.clip(quantum_mutant, self.lower_bound, self.upper_bound)
                    quantum_fitness = func(quantum_mutant)
                    self.evaluations += 1
                    if quantum_fitness < self.personal_best_scores[i]:
                        self.positions[i] = quantum_mutant
                        self.personal_best_scores[i] = quantum_fitness
                        self.personal_best_positions[i] = quantum_mutant

            # Enhanced mutation with dynamic contraction
            if np.random.rand() < self.adaptive_rate:
                for _ in range(int(self.population_size * 0.3)):  # Increased mutation rate
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    contraction_factor = 0.5 * (1 - self.evaluations / self.budget)
                    mutant_vector = self.positions[a] + contraction_factor * (self.positions[b] - self.positions[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    candidate_index = np.random.randint(0, self.population_size)
                    mutant_fitness = func(mutant_vector)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[candidate_index]:
                        self.positions[candidate_index] = mutant_vector
                        self.personal_best_scores[candidate_index] = mutant_fitness
                        self.personal_best_positions[candidate_index] = mutant_vector

        return self.global_best_score