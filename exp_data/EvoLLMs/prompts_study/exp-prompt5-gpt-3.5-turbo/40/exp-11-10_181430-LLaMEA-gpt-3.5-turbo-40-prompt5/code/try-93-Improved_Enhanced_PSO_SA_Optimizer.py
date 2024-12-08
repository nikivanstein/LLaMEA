import numpy as np

class Improved_Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.max_iter = 1200
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.6
        self.temp = 15.0
        self.alpha = 0.9
        self.mutation_rate = 0.1
        self.elitism_rate = 0.1
        
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        def accept_move(cost_diff, temp):
            return cost_diff < 0 or np.random.uniform(0, 1) < np.exp(-cost_diff / temp)
        
        # Initialize particles
        particles = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best = particles.copy()
        global_best = particles[np.argmin([objective_function(p) for p in particles])
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + self.c1 * r1 * (personal_best[i] - particles[i]) + self.c2 * r2 * (global_best - particles[i])
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                
                # Adaptive Mutation with Gaussian mutation
                for _ in range(5):
                    mutation_std = self.mutation_rate * np.std(particles)
                    new_particle = particles[i] + np.random.normal(0, mutation_std, size=self.dim)
                    cost_diff = objective_function(new_particle) - objective_function(particles[i])
                    if accept_move(cost_diff, self.temp):
                        particles[i] = new_particle
                
                # Elitism
                if np.random.uniform(0, 1) < self.elitism_rate:
                    particles[i] = personal_best[np.argmin([objective_function(p) for p in personal_best])].copy()
                    
                # Update personal best
                if objective_function(particles[i]) < objective_function(personal_best[i]):
                    personal_best[i] = particles[i].copy()
                # Update global best
                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()
                
            self.temp *= self.alpha

        return global_best