import numpy as np

class EnhancedNeighborhoodAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_weight_initial = 1.5
        self.social_weight_initial = 1.8
        self.mutation_scale = 0.2
        self.neighborhood_size = 5

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        dynamic_inertia = lambda iter: self.inertia_weight_initial - iter * (self.inertia_weight_initial - self.inertia_weight_final) / (self.budget/self.swarm_size)
        dynamic_cognitive = lambda iter: self.cognitive_weight_initial + iter * (2.0 - self.cognitive_weight_initial) / (self.budget/self.swarm_size)
        dynamic_social = lambda iter: self.social_weight_initial + iter * (2.5 - self.social_weight_initial) / (self.budget/self.swarm_size)
        
        iter_count = 0

        neighbors = lambda idx: np.random.choice(np.delete(np.arange(self.swarm_size), idx), self.neighborhood_size, replace=False)

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            for i in range(self.swarm_size):
                neighborhood_best = min(neighbors(i), key=lambda n: personal_best_scores[n])
                neighborhood_best_pos = personal_best_positions[neighborhood_best]

                velocities[i] = (dynamic_inertia(iter_count) * velocities[i] +
                                 dynamic_cognitive(iter_count) * r1 * (personal_best_positions[i] - positions[i]) +
                                 dynamic_social(iter_count) * r2 * (neighborhood_best_pos - positions[i]))
                
                velocities[i] = np.clip(velocities[i], -0.7, 0.7)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            scores = np.array([func(p) for p in positions])
            evaluations += self.swarm_size

            better_idxs = scores < personal_best_scores
            personal_best_positions[better_idxs] = positions[better_idxs]
            personal_best_scores[better_idxs] = scores[better_idxs]

            current_global_best_idx = np.argmin(personal_best_scores)
            current_global_best_score = personal_best_scores[current_global_best_idx]

            if current_global_best_score < global_best_score:
                global_best_position = personal_best_positions[current_global_best_idx]
                global_best_score = current_global_best_score

            iter_count += 1

        return global_best_position, global_best_score