import numpy as np

class EnhancedADSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, budget // 10)
        self.inertia_weight = 0.729
        self.cognitive_coef = 1.494
        self.social_coef = 1.494
        self.learning_rate = 0.05  # Adaptive learning rate for faster convergence
        self.momentum = 0.95      # Fine-tuned momentum for stability and exploration

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize particles with diverse distribution
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.normal(0, 0.1, (self.population_size, self.dim))  # Small perturbation for diversity
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        last_improvement = 0
        
        while eval_count < self.budget:
            # Evaluate current positions
            scores = np.array([func(x) for x in positions])
            eval_count += self.population_size
            
            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]
            
            # Update global best
            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]
                last_improvement = eval_count
            
            # Adjust inertia weight based on improvement
            if eval_count - last_improvement > self.population_size:
                self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
            
            # Dynamic adjustment of coefficients
            self.cognitive_coef = 1.494 + 0.5 * (1 - eval_count / self.budget)
            self.social_coef = 1.494 - 0.5 * (1 - eval_count / self.budget)
            
            # Adaptive learning rate scaling
            adaptive_lr = self.learning_rate * (1.0 + (0.5 * (self.budget - eval_count) / self.budget))
            
            # Update velocities and positions with adjusted momentum and learning rate
            r1, r2 = np.random.uniform(size=(2, self.population_size, self.dim))
            cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coef * r2 * (global_best_position - positions)
            velocities = (self.inertia_weight * velocities + cognitive_velocity + social_velocity)
            velocities = self.momentum * velocities + adaptive_lr * (cognitive_velocity + social_velocity)
            
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_score