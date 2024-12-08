import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def chaotic_map(self, iteration):
        return 0.7 * (1 - (iteration / self.budget)) * np.sin(10 * np.pi * iteration / self.budget)

    def __call__(self, func):
        F = 0.5  # Differential evolution mutation factor
        CR = 0.9  # Crossover probability

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate the fitness of the particle
                fitness = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            # Calculate adaptive inertia weight using chaotic map
            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * 
                                                        (self.evaluations / self.budget))
            chaotic_factor = self.chaotic_map(self.evaluations)
            inertia_weight *= (1 + chaotic_factor)

            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])

                # Differential Evolution mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = self.positions[a] + F * (self.positions[b] - self.positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                crossover_vector = np.array([mutant_vector[j] if np.random.rand() < CR else self.positions[i][j] for j in range(self.dim)])
                
                self.velocities[i] = (inertia_weight * self.velocities[i] + 
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i] * (1 + chaotic_factor)
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                # Apply crossover
                self.positions[i] = crossover_vector

        return self.global_best_score