import numpy as np
from minisom import MiniSom

class EnhancedHybridPSODESOM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 15
        self.inertia_weight = 0.9  # Start higher for exploration
        self.inertia_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1
        self.som = MiniSom(1, self.num_particles, self.dim, sigma=0.5, learning_rate=0.5)

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.num_particles, self.dim))
        p_best = particles.copy()
        p_best_values = np.array([func(p) for p in particles])
        g_best = p_best[np.argmin(p_best_values)]
        g_best_value = np.min(p_best_values)

        eval_count = self.num_particles

        while eval_count < self.budget:
            # Adaptive inertia weight
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (eval_count / self.budget))
            
            for i in range(self.num_particles):
                # Update velocity and position (PSO)
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] 
                                 + self.c1 * r1 * (p_best[i] - particles[i]) 
                                 + self.c2 * r2 * (g_best - particles[i]))
                # Limit velocities
                velocities[i] = np.clip(velocities[i], -self.vel_max, self.vel_max)
                particles[i] += velocities[i]
                # Boundary check
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Evaluate particle
                value = func(particles[i])
                eval_count += 1

                # Update personal best
                if value < p_best_values[i]:
                    p_best[i] = particles[i].copy()
                    p_best_values[i] = value

                # Update global best
                if value < g_best_value:
                    g_best = particles[i].copy()
                    g_best_value = value

                if eval_count >= self.budget:
                    break

            # Apply DE mutation and crossover
            if eval_count < self.budget:
                for i in range(self.num_particles):
                    indices = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    
                    mutant = p_best[a] + self.F * (p_best[b] - p_best[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, particles[i])
                    
                    trial_value = func(trial)
                    eval_count += 1
                    
                    if trial_value < p_best_values[i]:
                        p_best[i] = trial
                        p_best_values[i] = trial_value
                        
                        if trial_value < g_best_value:
                            g_best = trial
                            g_best_value = trial_value
                            
                    if eval_count >= self.budget:
                        break

            # SOM adaptation
            self.som.train_random(particles, 10)
            for i, particle in enumerate(particles):
                particles[i] = self.som.winner(particle)

        return g_best, g_best_value