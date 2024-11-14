import numpy as np

class Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_iter = 1000
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.5
        self.temp = 10.0
        self.alpha = 0.95
        self.threshold = 0.1
        self.collapse_factor = 0.5
        
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        def accept_move(cost_diff, temp):
            if cost_diff < 0:
                return True
            return np.random.uniform(0, 1) < np.exp(-cost_diff / temp)
        
        # Initialize particles
        pop_size = 10
        particles = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        velocities = np.zeros((pop_size, self.dim))
        personal_best = particles.copy()
        global_best = particles[np.argmin([objective_function(p) for p in particles])]
        
        for _ in range(self.max_iter):
            for i in range(pop_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + self.c1 * r1 * (personal_best[i] - particles[i]) + self.c2 * r2 * (global_best - particles[i])
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                
                # Simulated Annealing
                for _ in range(5):
                    new_particle = particles[i] + np.random.normal(0, self.temp, size=self.dim)
                    cost_diff = objective_function(new_particle) - objective_function(particles[i])
                    if accept_move(cost_diff, self.temp):
                        particles[i] = new_particle
                
                # Update personal best
                if objective_function(particles[i]) < objective_function(personal_best[i]):
                    personal_best[i] = particles[i].copy()
                # Update global best
                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()
            
            self.temp *= self.alpha
            
            # Population size adaptation
            if np.random.uniform(0, 1) < self.threshold:
                pop_size = max(int(pop_size * self.collapse_factor), 2)
                particles = np.vstack((particles, np.random.uniform(-5.0, 5.0, size=(pop_size-particles.shape[0], self.dim))))
                velocities = np.vstack((velocities, np.zeros((pop_size-velocities.shape[0], self.dim)))
                personal_best = np.vstack((personal_best, particles[-pop_size:].copy()))
        
        return global_best