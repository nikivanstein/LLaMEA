import numpy as np

class PSO_ALS_DP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.differential_weight = 0.8
        self.adaptive_weight_decay = 0.9
        
    def __call__(self, func):
        np.random.seed(0)
        
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evals = self.pop_size
        
        while evals < self.budget:
            for i in range(self.pop_size):
                # Update velocities and positions with adaptive inertia
                self.inertia_weight *= self.adaptive_weight_decay
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                                 + self.social_coeff * r2 * (global_best_position - positions[i]))
                
                # Apply differential perturbation
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                diff_perturbation = self.differential_weight * (positions[idxs[0]] - positions[idxs[1]])
                velocities[i] += diff_perturbation
                
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new positions
                score = func(positions[i])
                evals += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
            
            # Adaptive Local Search with differential perturbation
            if evals < self.budget:
                local_positions = global_best_position + np.random.uniform(-0.1, 0.1, self.dim) + diff_perturbation
                local_positions = np.clip(local_positions, self.lower_bound, self.upper_bound)
                local_score = func(local_positions)
                evals += 1
                if local_score < global_best_score:
                    global_best_score = local_score
                    global_best_position = local_positions
        
        return global_best_score