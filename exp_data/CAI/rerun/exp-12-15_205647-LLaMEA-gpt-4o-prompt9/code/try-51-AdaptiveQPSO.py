import numpy as np

class AdaptiveQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(60, budget // 8)
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.quantum_coef = 0.5
        self.shrink_weight = 0.9

    def __call__(self, func):
        np.random.seed(42)
        
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        last_improvement = 0

        while eval_count < self.budget:
            scores = np.array([func(x) for x in positions])
            eval_count += self.population_size
            
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]
            
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]
                last_improvement = eval_count
            
            if eval_count - last_improvement > 2 * self.population_size:
                self.population_size = max(20, int(self.population_size * self.shrink_weight))
                positions = positions[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]
            
            r1, r2 = np.random.uniform(size=(2, self.population_size, self.dim))
            cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coef * r2 * (global_best_position - positions)

            # Quantum-inspired potential field
            quantum_position = global_best_position + self.quantum_coef * np.random.uniform(-1, 1, self.dim)
            quantum_velocity = self.quantum_coef * r1 * (quantum_position - positions)

            velocities = self.inertia_weight * velocities + cognitive_velocity + social_velocity + quantum_velocity
            
            max_velocity = 0.3 * (self.upper_bound - self.lower_bound)
            velocities = np.clip(velocities, -max_velocity, max_velocity)
            
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_score