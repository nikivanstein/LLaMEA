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
        self.learning_rate = 0.05
        self.momentum = 0.95
        self.f_scale = 0.5  # Scaling factor for differential evolution

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
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
            
            # Adjust inertia weight and velocity scaling
            if eval_count - last_improvement > self.population_size:
                self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
            
            # Dynamic adjustment of coefficients with scaling factor
            self.cognitive_coef = (1.494 + 0.5 * (1 - eval_count / self.budget)) * self.f_scale
            self.social_coef = (1.494 - 0.5 * (1 - eval_count / self.budget)) * self.f_scale
            
            # Update velocities and positions with differential evolution crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant_vector = np.clip(a + self.f_scale * (b - c), self.lower_bound, self.upper_bound)
                r = np.random.rand(self.dim)
                trial_vector = np.where(r < 0.5, mutant_vector, positions[i])
                
                r1, r2 = np.random.uniform(size=(2, self.dim))
                cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions[i] - trial_vector)
                social_velocity = self.social_coef * r2 * (global_best_position - trial_vector)
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity)
                velocities[i] = self.momentum * velocities[i] + self.learning_rate * (cognitive_velocity + social_velocity)
                
                positions[i] = np.clip(trial_vector + velocities[i], self.lower_bound, self.upper_bound)
        
        return global_best_position, global_best_score