import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(60, 14 * dim)  # Adjusted number of particles for dynamic exploration
        self.c1 = 1.5  # Adjusted cognitive component for adaptive learning
        self.c2 = 2.0  # Slightly reduced social component to balance global influence
        self.w = 0.9  # Increased inertia weight for initial exploration
        self.decay_rate = 0.95  # Adjusted decay rate for more gradual change in inertia
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.12  # Adjusted velocity limit for strategic movement
    
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
            
            if evaluations < self.budget * 0.5:
                self.c1 += 0.05  # Gradually increase learning to adapt exploration
                self.c2 -= 0.05  # Gradually decrease social influence to refine individual search
            
            self.w *= self.decay_rate  # Decay inertia weight
        
        return global_best_position, global_best_score