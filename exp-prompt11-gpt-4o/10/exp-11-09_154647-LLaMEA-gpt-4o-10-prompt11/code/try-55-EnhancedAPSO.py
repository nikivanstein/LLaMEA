import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_num_particles = min(70, 12 * dim)  # Initial number of particles
        self.c1 = 1.5  # Adjusted cognitive component for improved balance
        self.c2 = 2.0  # Adjusted social component for better collaboration
        self.w = 0.80  # Adjusted inertia weight for improved convergence
        self.decay_rate = 0.95  # Slower decay for sustained exploration
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.15  # Increased velocity limit for dynamic responses
    
    def __call__(self, func):
        np.random.seed(42)
        num_particles = self.initial_num_particles
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(pos)
        personal_best_scores = np.array([func(p) for p in pos])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = num_particles
        
        while evaluations < self.budget:
            # Adaptively adjust swarm size
            if evaluations % (self.budget // 5) == 0:
                num_particles = min(num_particles + 5, self.budget - evaluations)

            for i in range(num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                # Adaptive velocity calculation
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
            
            self.w *= self.decay_rate  # Decay inertia weight
        
        return global_best_position, global_best_score