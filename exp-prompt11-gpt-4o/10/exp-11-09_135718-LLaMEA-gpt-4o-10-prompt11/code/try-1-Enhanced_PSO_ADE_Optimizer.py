import numpy as np

class Enhanced_PSO_ADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50
        self.max_mutation_factor = 0.9
        self.min_mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.max_inertia_weight = 0.9
        self.min_inertia_weight = 0.4
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        eval_count = 0
        iteration = 0

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

            r1, r2 = np.random.rand(2)
            inertia_weight = self.max_inertia_weight - (self.max_inertia_weight - self.min_inertia_weight) * (eval_count / self.budget)
            velocities = (
                inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.swarm_size):
                indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                x0, x1, x2 = positions[indices]
                mutation_factor = self.max_mutation_factor - (self.max_mutation_factor - self.min_mutation_factor) * (eval_count / self.budget)
                mutated_vector = np.clip(x0 + mutation_factor * (x1 - x2), self.lower_bound, self.upper_bound)
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