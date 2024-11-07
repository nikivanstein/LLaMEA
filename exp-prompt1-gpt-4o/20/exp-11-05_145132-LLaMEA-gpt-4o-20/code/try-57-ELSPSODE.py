import numpy as np

class ELSPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.w_init = 0.9  # Initial inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.F = 0.8  # Scaling factor for DE
        self.CR = 0.9  # Crossover probability for DE

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility

        # Initialize particle positions and velocities for PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Adaptive inertia weight
            w = self.w_min + (self.w_init - self.w_min) * (1 - evaluations / self.budget)

            # PSO Update
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities + self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            # Evaluate new positions
            scores = np.array([func(p) for p in positions])
            evaluations += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = positions[i]

            # Local Search Phase
            if evaluations < self.budget:
                for i in range(self.population_size):
                    local_mutation = np.random.uniform(-0.1, 0.1, self.dim)
                    local_vector = np.clip(personal_best_positions[i] + local_mutation, self.lower_bound, self.upper_bound)
                    local_score = func(local_vector)
                    evaluations += 1
                    if local_score < personal_best_scores[i]:
                        personal_best_scores[i] = local_score
                        personal_best_positions[i] = local_vector
                        if local_score < global_best_score:
                            global_best_score = local_score
                            global_best_position = local_vector

            if evaluations >= self.budget:
                break

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                if i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, positions[i])

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < scores[i]:
                    positions[i] = trial_vector
                    scores[i] = trial_score

                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score