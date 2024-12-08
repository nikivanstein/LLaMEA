import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnhancedPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.f = 0.8
        self.cr = 0.9
        self.forest = RandomForestRegressor(n_estimators=10, random_state=42)

    def __call__(self, func):
        np.random.seed(42)
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.pop_size
        samples = np.copy(pop)
        scores = np.copy(personal_best_scores)

        while eval_count < self.budget:
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop += velocities
            pop = np.clip(pop, self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in pop])
            eval_count += self.pop_size
            
            samples = np.concatenate((samples, pop), axis=0)
            scores = np.concatenate((scores, personal_best_scores), axis=0)
            self.forest.fit(samples, scores)
            predicted_scores = self.forest.predict(pop)
            better_mask = predicted_scores < personal_best_scores
            
            personal_best_scores = np.where(better_mask, predicted_scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], pop, personal_best_positions)

            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_idx]
                global_best_position = personal_best_positions[current_global_best_idx]
            
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = personal_best_positions[a] + self.f * (personal_best_positions[b] - personal_best_positions[c])
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