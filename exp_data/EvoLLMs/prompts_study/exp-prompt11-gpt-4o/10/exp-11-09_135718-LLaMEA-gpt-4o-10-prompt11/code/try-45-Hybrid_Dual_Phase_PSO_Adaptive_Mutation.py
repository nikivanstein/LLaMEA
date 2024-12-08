import numpy as np

class Hybrid_Dual_Phase_PSO_Adaptive_Mutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50
        self.mutation_factor = 0.85  # Adjusted mutation factor
        self.crossover_prob = 0.9
        self.inertia_weight = 0.6  # Increased inertia_weight for exploration
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.4  # Slightly increased social_coeff
        self.adaptive_mutation_phase = budget // 4  # New dual-phase approach

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        eval_count = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            eval_count += self.swarm_size

            for i in range(self.swarm_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]

                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]

            # Adjust inertia_weight dynamically
            self.inertia_weight *= 0.97  # More gradual reduction

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.swarm_size):
                if eval_count < self.adaptive_mutation_phase:
                    indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                    x0, x1, x2 = positions[indices]
                    mutated_vector = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lower_bound, self.upper_bound)
                else:
                    mutated_vector = np.clip(personal_best_positions[i] + self.mutation_factor * (global_best_position - personal_best_positions[i]), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutated_vector, positions[i])
                
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_score