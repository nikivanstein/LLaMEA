import numpy as np

class ReinforcedHybridSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30  # Increased population size for diversity
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))  # Reduced velocity range
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.mutation_factor = 0.9  # Slightly increased for enhanced exploration
        self.crossover_rate = 0.7  # Increased crossover rate for more genetic diversity
        self.c1 = 2.0  # Balanced cognitive factor
        self.c2 = 2.0  # Balanced social factor
        self.inertia_weight = 0.4  # Lower inertia for faster convergence
        self.max_evaluations = budget
        self.cooling_factor = 0.990  # More aggressive cooling factor
        self.exploration_factor = 0.25  # Further increased for diversity
        self.adaptive_mutation = 0.07  # Slightly increased adaptive mutation rate
        self.exploration_phase = True
        self.contextual_learning_rate = 0.05  # New parameter for contextual learning

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                if evaluations >= self.max_evaluations:
                    break

            for i in range(self.population_size):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if self.exploration_phase:
                for i in range(self.population_size):
                    indices = [index for index in range(self.population_size) if index != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = self.particles[a] + self.mutation_factor * (self.particles[b] - self.particles[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    trial_vector = np.copy(self.particles[i])
                    if np.random.rand() < self.crossover_rate:
                        trial_vector = (1 - self.adaptive_mutation) * trial_vector + self.adaptive_mutation * mutant_vector
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.particles[i] = trial_vector
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector
                    if evaluations >= self.max_evaluations:
                        break

            if not self.exploration_phase:
                diversity = np.mean(np.std(self.particles, axis=0))
                adaptive_step_size = self.exploration_factor * diversity
                for i in range(self.population_size):
                    if np.random.rand() < 0.4:  # Increased random step frequency
                        random_step = np.random.normal(0, adaptive_step_size, self.dim)
                        candidate_vector = np.clip(self.particles[i] + random_step, self.lower_bound, self.upper_bound)
                        candidate_score = func(candidate_vector)
                        evaluations += 1
                        if candidate_score < self.personal_best_scores[i]:
                            self.particles[i] = candidate_vector
                            self.personal_best_scores[i] = candidate_score
                            self.personal_best_positions[i] = candidate_vector
                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_vector
                        if evaluations >= self.max_evaluations:
                            break
            
            if evaluations % 100 == 0:
                self.exploration_phase = not self.exploration_phase
                if not self.exploration_phase:
                    # Implement a contextual learning step to enhance adaptation
                    mean_position = np.mean(self.particles, axis=0)
                    self.global_best_position = self.global_best_position + self.contextual_learning_rate * (mean_position - self.global_best_position)
            
            self.mutation_factor *= self.cooling_factor
            self.inertia_weight *= self.cooling_factor
            self.exploration_factor *= self.cooling_factor
            self.adaptive_mutation *= self.cooling_factor
        
        return self.global_best_position, self.global_best_score