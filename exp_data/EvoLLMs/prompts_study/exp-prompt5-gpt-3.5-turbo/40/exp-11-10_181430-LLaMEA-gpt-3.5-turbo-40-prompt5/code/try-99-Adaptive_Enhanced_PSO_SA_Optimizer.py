import numpy as np

class Adaptive_Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.max_iter = 1200
        self.c1 = 1.5
        self.c2 = 1.5
        self.initial_inertia_weight = 0.9
        self.temp = 15.0
        self.alpha = 0.9
        self.lbound = -5.0
        self.ubound = 5.0
        
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        def accept_move(cost_diff, temp):
            return cost_diff < 0 or np.random.uniform(0, 1) < np.exp(-cost_diff / temp)
        
        # Initialize particles
        particles = np.random.uniform(self.lbound, self.ubound, size=(self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best = particles.copy()
        global_best = particles[np.argmin([objective_function(p) for p in particles])
        inertia_weight = self.initial_inertia_weight

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = inertia_weight * velocities[i] + self.c1 * r1 * (personal_best[i] - particles[i]) + self.c2 * r2 * (global_best - particles[i])
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], self.lbound, self.ubound)
                
                # Simulated Annealing with Gaussian mutation
                for _ in range(5):
                    mutation_rate = 1.0 / np.sqrt(1 + inertia_weight)
                    new_particle = particles[i] + mutation_rate * np.random.normal(0, self.temp, size=self.dim)
                    cost_diff = objective_function(new_particle) - objective_function(particles[i])
                    if accept_move(cost_diff, self.temp):
                        particles[i] = new_particle
                
                # Update personal best
                if objective_function(particles[i]) < objective_function(personal_best[i]):
                    personal_best[i] = particles[i].copy()
                # Update global best
                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()
                    
            inertia_weight = self.update_inertia(inertia_weight)

        return global_best
      
    def update_inertia(self, inertia_weight):
        return max(0.3, inertia_weight * 0.995)  # Reduce inertia weight gradually