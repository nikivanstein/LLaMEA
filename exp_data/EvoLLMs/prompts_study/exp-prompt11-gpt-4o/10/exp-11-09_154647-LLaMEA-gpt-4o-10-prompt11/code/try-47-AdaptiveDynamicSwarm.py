import numpy as np

class AdaptiveDynamicSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(80, 13 * dim)  # Slightly increased for more exploration
        self.c1_ini = 1.5  # Initial cognitive component
        self.c2_ini = 2.0  # Initial social component
        self.c1_final = 1.0  # Final cognitive component for precision
        self.c2_final = 2.5  # Final social component for convergence
        self.w = 0.9  # Increased initial inertia for exploration
        self.w_min = 0.4  # Minimum inertia weight for exploitation
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.15  # Increase for more reach

    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(pos)
        personal_best_scores = np.array([func(p) for p in pos])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.num_particles

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            self.c1 = self.c1_ini + progress_ratio * (self.c1_final - self.c1_ini)
            self.c2 = self.c2_ini + progress_ratio * (self.c2_final - self.c2_ini)
            self.w = self.w - progress_ratio * (self.w - self.w_min)

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                vel[i] = (self.w * vel[i] + 
                          self.c1 * r1 * (personal_best_positions[i] - pos[i]) + 
                          self.c2 * r2 * (global_best_position - pos[i]))
                
                vel[i] = np.clip(vel[i], -self.velocity_limit, self.velocity_limit)
                pos[i] += vel[i]
                pos[i] = np.clip(pos[i], self.lower_bound, self.upper_bound)
                
                score = func(pos[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pos[i]
                
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pos[i]
        
        return global_best_position, global_best_score