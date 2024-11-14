import numpy as np

class LevyFlightParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (5 * dim)))  # heuristic for population size
        self.w = 0.7  # inertia weight
        self.c1 = 1.5 # cognitive (particle) constant
        self.c2 = 1.5 # social (swarm) constant
        
    def levy_flight(self, size, beta=1.5):
        # Lévy flight step-size generation using Mantegna's algorithm
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step
    
    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_fitness = np.array([func(ind) for ind in particles])
        num_evaluations = self.population_size
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocity using inertia, personal and global best
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) + 
                                 self.c2 * r2 * (global_best_position - particles[i]))
                
                # Levy flight exploration
                levy_step = self.levy_flight(self.dim)
                velocities[i] += levy_step
                
                # Update particle position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)
                
                # Evaluate fitness
                fitness = func(particles[i])
                num_evaluations += 1
                
                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best_position = particles[i]
                        global_best_fitness = fitness
        
        return global_best_position, global_best_fitness