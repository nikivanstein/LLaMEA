import numpy as np

class EnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = min(80, 14 * dim)  # Increased particles for initial exploration
        self.c1 = 1.2  # Further reduced cognitive component for enhanced social learning
        self.c2 = 2.5  # Increased social component for stronger convergence towards global best
        self.w = 0.9  # Adjusted inertia weight for initial rapid exploration
        self.decay_rate = 0.91  # Faster decay to quickly switch to exploitation
        self.velocity_limit = (self.upper_bound - self.lower_bound) * 0.15  # Increased velocity for initial phase
    
    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(pos)
        personal_best_scores = np.array([func(p) for p in pos])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.num_particles
        
        dynamic_threshold = int(self.budget * 0.5)  # Switch strategy halfway through the budget

        while evaluations < self.budget:
            if evaluations > dynamic_threshold:
                self.c1, self.c2 = 0.5, 3.0  # Shift to exploitation phase with more social influence
                self.num_particles = max(50, 10 * self.dim)  # Reduce particles for focused search
            
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
        
        return global_best_position, global_best_score