import numpy as np

class AdaptiveHybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 + 3 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.65  # Differential Evolution parameter, increased from 0.6 to 0.65
        self.CR = 0.97  # Crossover probability, increased from 0.95 to 0.97
        self.w = 0.45  # Inertia weight for PSO, increased from 0.4 to 0.45
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient

    def __call__(self, func):
        np.random.seed(42)
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_pos = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in personal_best_pos])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_pos = personal_best_pos[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        evaluations = self.pop_size

        while evaluations < self.budget:
            # Differential Evolution Step
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                
                x1, x2, x3 = pop[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(pop[i])
                
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]
                        
                trial_score = func(trial)
                evaluations += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_pos[i] = trial
                
                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_pos = trial
                
            # Particle Swarm Optimization Step
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # Modifying the cognitive and social coefficients with time-varying factors
                self.c1 = 1.5 + evaluations/self.budget * 1.2  # Adjusted cognitive coefficient dynamics
                self.c2 = 0.5 + evaluations/self.budget * 1.5
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_pos[i] - pop[i]) +
                                 self.c2 * r2 * (global_best_pos - pop[i]))
                pop[i] = pop[i] + velocities[i]
                pop[i] = np.clip(pop[i], self.lower_bound, self.upper_bound)
                
                score = func(pop[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_pos[i] = pop[i]
                
                if score < global_best_score:
                    global_best_score = score
                    global_best_pos = pop[i]
        
        return global_best_pos, global_best_score