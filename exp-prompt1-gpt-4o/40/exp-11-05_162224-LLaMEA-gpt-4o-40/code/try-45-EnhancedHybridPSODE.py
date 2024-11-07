import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 25  # Increased population size for diversity
        self.c1 = 2.0  # Enhanced PSO cognitive coefficient
        self.c2 = 1.5  # PSO social coefficient
        self.w = 0.5  # Reduced inertia weight for faster convergence
        self.f_min = 0.5  # Dynamic DE scaling factor
        self.f_max = 0.9
        self.cr = 0.8  # Slightly reduced DE crossover probability

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop += velocities
            pop = np.clip(pop, self.lower_bound, self.upper_bound)
            
            # Evaluate the population
            scores = np.array([func(ind) for ind in pop])
            eval_count += self.pop_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], pop, personal_best_positions)
            
            # Update global best
            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_idx]
                global_best_position = personal_best_positions[current_global_best_idx]
            
            # Differential Evolution Update with adaptive scaling factor
            f_dynamic = self.f_min + (self.f_max - self.f_min) * (eval_count / self.budget)
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = personal_best_positions[a] + f_dynamic * (personal_best_positions[b] - personal_best_positions[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                trial_score = func(trial)
                eval_count += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                
                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial
                    
                if eval_count >= self.budget:
                    break
        
        return global_best_position, global_best_score