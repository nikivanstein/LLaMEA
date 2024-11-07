import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.num_swarms = 2  # Added multiswarm capability
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        np.random.seed(42)

        # Initialize swarms
        swarms = [np.random.uniform(self.lower_bound, self.upper_bound, 
                                    (self.population_size // self.num_swarms, self.dim)) 
                  for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, s.shape) for s in swarms]
        personal_best_positions = [np.copy(s) for s in swarms]
        personal_best_scores = [np.array([func(p) for p in s]) for s in personal_best_positions]
        
        global_best_position = min([pb[np.argmin(pbs)] for pb, pbs in zip(personal_best_positions, personal_best_scores)], 
                                   key=lambda p: func(p))
        global_best_score = func(global_best_position)

        evaluations = sum(len(pb) for pb in personal_best_scores)
        
        while evaluations < self.budget:
            for swarm_index, (positions, velocity, personal_best_position, personal_best_score) in enumerate(zip(swarms, velocities, personal_best_positions, personal_best_scores)):
                r1, r2 = np.random.rand(2, *positions.shape)
                velocity = (self.w * velocity + self.c1 * r1 * (personal_best_position - positions) +
                            self.c2 * r2 * (global_best_position - positions))
                positions = np.clip(positions + velocity, self.lower_bound, self.upper_bound)

                # Evaluate new positions
                scores = np.array([func(p) for p in positions])
                evaluations += len(positions)

                # Update personal and global bests
                for i in range(len(positions)):
                    if scores[i] < personal_best_score[i]:
                        personal_best_score[i] = scores[i]
                        personal_best_position[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

                # DE Mutation and Crossover with adaptive F and CR
                self.F = 0.5 + np.random.rand() * 0.5  # Adaptive F
                self.CR = 0.8 + np.random.rand() * 0.2  # Adaptive CR
                for i in range(len(positions)):
                    indices = np.random.choice(len(positions), 3, replace=False)
                    x1, x2, x3 = positions[indices]
                    mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)

                    crossover_mask = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover_mask, mutant_vector, positions[i])

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < scores[i]:
                        positions[i] = trial_vector
                        scores[i] = trial_score

                        if trial_score < personal_best_score[i]:
                            personal_best_score[i] = trial_score
                            personal_best_position[i] = trial_vector
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

                    if evaluations >= self.budget:
                        break

        return global_best_position, global_best_score