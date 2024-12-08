import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(60, 15 * dim)
        self.c1 = 1.6  # Slightly increased cognitive component for better individual learning
        self.c2 = 1.9  # Reduced social component to encourage diverse solutions
        self.w_max = 0.9  # Adjusted inertia weight range for dynamic control
        self.w_min = 0.4
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.07
        self.mutation_prob = 0.1  # Introduce mutation probability for velocity

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
            current_iter = evaluations // self.num_particles
            w = self.w_max - (self.w_max - self.w_min) * (current_iter / (self.budget // self.num_particles))
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                vel[i] = (w * vel[i] + 
                          self.c1 * r1 * (personal_best_positions[i] - pos[i]) + 
                          self.c2 * r2 * (global_best_position - pos[i]))
                
                if np.random.rand() < self.mutation_prob:
                    mutation = np.random.normal(0, 0.1, self.dim)  
                    vel[i] += mutation
                
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