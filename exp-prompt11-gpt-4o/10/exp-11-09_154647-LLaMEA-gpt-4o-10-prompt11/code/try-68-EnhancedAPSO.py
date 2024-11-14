import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(75, 14 * dim)  # Adjusted particles for exploration
        self.c1 = 1.3  # Reduced cognitive component for better global search
        self.c2 = 2.4  # Increased social component for stronger collective behavior
        self.w = 0.9  # Slightly increased inertia weight for initial exploration
        self.decay_rate = 0.92  # Faster decay for quicker exploitation transition
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.12  # Increased velocity limit for dynamic movement
        self.dynamic_adjustment_factor = 0.05  # Factor to adjust particles dynamically
    
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
            
            self.w *= self.decay_rate  # Decay inertia weight
            # Dynamic adjustment of particles based on convergence
            if evaluations < self.budget * 0.5 and np.random.rand() < self.dynamic_adjustment_factor:
                new_particle_pos = np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))
                pos = np.vstack((pos, new_particle_pos))
                vel = np.vstack((vel, np.random.uniform(-1, 1, (1, self.dim))))
                new_particle_score = func(new_particle_pos[0])
                personal_best_positions = np.vstack((personal_best_positions, new_particle_pos))
                personal_best_scores = np.append(personal_best_scores, new_particle_score)
                if new_particle_score < global_best_score:
                    global_best_score = new_particle_score
                    global_best_position = new_particle_pos[0]
                evaluations += 1
                self.num_particles += 1
        
        return global_best_position, global_best_score