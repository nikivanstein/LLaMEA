import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(80, 10 * dim)  # Adjusted particles for dynamic exploration
        self.c1 = 1.5  # Adjusted cognitive component for better individual exploration
        self.c2 = 2.1  # Slightly reduced social component for more controlled convergence
        self.w = 0.8  # Further reduced inertia weight for quicker convergence
        self.decay_rate = 0.95  # Slowed decay to maintain momentum longer
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.15  # Increased velocity limit for more aggressive movement
        self.max_swarm_size = min(100, 15 * dim)  # New dynamic maximum swarm size
    
    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(pos)
        personal_best_scores = np.array([func(p) for p in pos])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.num_particles
        adaptive_c1 = self.c1  # Introduce adaptive c1
        adaptive_c2 = self.c2  # Introduce adaptive c2
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                vel[i] = (self.w * vel[i] + 
                          adaptive_c1 * r1 * (personal_best_positions[i] - pos[i]) + 
                          adaptive_c2 * r2 * (global_best_position - pos[i]))
                
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
            
            adaptive_c1 = self.c1 * (1 - evaluations / self.budget)  # Decrease c1 over time
            adaptive_c2 = self.c2 * (1 + evaluations / self.budget)  # Increase c2 over time
            self.w *= self.decay_rate  # Decay inertia weight
            
            # Dynamic swarm size adjustment
            if evaluations < self.budget * 0.5:
                self.num_particles = min(self.max_swarm_size, self.num_particles + 1)
        
        return global_best_position, global_best_score