import numpy as np

class AdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20  # Population size
        self.init_c1 = 1.5  # Initial PSO cognitive coefficient
        self.init_c2 = 1.5  # Initial PSO social coefficient
        self.init_w = 0.7   # Initial inertia weight
        self.init_f = 0.8   # Initial DE scaling factor
        self.init_cr = 0.9  # Initial DE crossover probability
        self.final_c1 = 1.0  # Final PSO cognitive coefficient
        self.final_c2 = 2.0  # Final PSO social coefficient
        self.final_w = 0.4   # Final inertia weight
        self.final_f = 0.5   # Final DE scaling factor
        self.final_cr = 0.6  # Final DE crossover probability

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
            # Adaptation of parameters
            progress = eval_count / self.budget
            c1 = self.init_c1 + progress * (self.final_c1 - self.init_c1)
            c2 = self.init_c2 + progress * (self.final_c2 - self.init_c2)
            w = self.init_w + progress * (self.final_w - self.init_w)
            f = self.init_f + progress * (self.final_f - self.init_f)
            cr = self.init_cr + progress * (self.final_cr - self.init_cr)

            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          c1 * r1 * (personal_best_positions - pop) +
                          c2 * r2 * (global_best_position - pop))
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
            
            # Differential Evolution Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = personal_best_positions[a] + f * (personal_best_positions[b] - personal_best_positions[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < cr, mutant, pop[i])
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