import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.dynamic_mutation_factor = 0.05

    def gaussian_mutation(self, size):
        return np.random.normal(0, 1, size)

    def update_population_size(self):
        new_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
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
                    break

                fitness = func(self.positions[i])
                self.evaluations += 1

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            inertia_weight = self.inertia_weight * (self.budget - self.evaluations) / self.budget

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Dynamic mutation with Gaussian perturbation
            if np.random.rand() < self.dynamic_mutation_factor:
                for i in range(self.population_size):
                    mutation_step = self.gaussian_mutation(self.dim)
                    mutant = self.positions[i] + 0.1 * mutation_step
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_fitness = func(mutant)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[i]:
                        self.positions[i] = mutant
                        self.personal_best_scores[i] = mutant_fitness
                        self.personal_best_positions[i] = mutant

            # Local search strategy
            if np.random.rand() < self.dynamic_mutation_factor:
                for i in range(self.population_size):
                    local_best = np.copy(self.positions[i])
                    local_best_score = func(local_best)
                    self.evaluations += 1
                    for _ in range(5):  # Fixed number of local iterations
                        if self.evaluations >= self.budget:
                            break
                        candidate = local_best + np.random.uniform(-0.1, 0.1, self.dim)
                        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                        candidate_score = func(candidate)
                        self.evaluations += 1
                        if candidate_score < local_best_score:
                            local_best, local_best_score = candidate, candidate_score

                    if local_best_score < self.personal_best_scores[i]:
                        self.positions[i] = local_best
                        self.personal_best_scores[i] = local_best_score
                        self.personal_best_positions[i] = local_best

        return self.global_best_score