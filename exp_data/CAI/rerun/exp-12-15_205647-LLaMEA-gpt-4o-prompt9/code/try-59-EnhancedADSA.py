import numpy as np

class EnhancedADSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, budget // 10)
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.729
        self.cognitive_coef = 1.494
        self.social_coef = 1.494
        self.learning_rate = 0.05
        self.momentum = 0.95
        self.local_search_intensity = 0.1

    def __call__(self, func):
        np.random.seed(42)
        
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
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

            if eval_count - last_improvement > self.population_size:
                self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
                self.population_size = max(10, int(self.population_size * 0.9))
                positions = positions[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]
                
                # Inject diversity with random individuals
                if np.random.rand() > 0.75:
                    diversity_positions = np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))
                    positions[-5:] = diversity_positions

            self.cognitive_coef = 1.494 + 0.3 * (1 - eval_count / self.budget)
            self.social_coef = 1.494 - 0.3 * (1 - eval_count / self.budget)
            self.learning_rate = 0.05 + 0.65 * (eval_count / self.budget)
            
            r1, r2 = np.random.uniform(size=(2, self.population_size, self.dim))
            cognitive_velocity = self.cognitive_coef * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coef * r2 * (global_best_position - positions)
            velocities = self.inertia_weight * velocities + cognitive_velocity + social_velocity
            velocities = self.momentum * velocities + self.learning_rate * (cognitive_velocity + social_velocity)
            
            max_velocity = 0.25 * (self.upper_bound - self.lower_bound)
            velocities = np.clip(velocities, -max_velocity, max_velocity)
            
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # Local search for best individual
            local_search_candidate = global_best_position + self.local_search_intensity * np.random.normal(size=self.dim)
            local_search_candidate = np.clip(local_search_candidate, self.lower_bound, self.upper_bound)
            local_search_score = func(local_search_candidate)
            eval_count += 1
            if local_search_score < global_best_score:
                global_best_score = local_search_score
                global_best_position = local_search_candidate
        
        return global_best_position, global_best_score