import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        n_particles = 20
        n_parents = 2
        F = 0.5
        CR = 0.7
        max_iter = self.budget // n_particles
        alpha = 0.9
        T0 = 1.0
        T_min = 1e-5
        
        def acceptance_probability(cost, new_cost, T):
            if new_cost < cost:
                return 1.0
            return np.exp((cost - new_cost) / T)
        
        # Initialize particles
        particles = np.random.uniform(-5.0, 5.0, size=(n_particles, self.dim))
        best_particle = particles[np.argmin([objective_function(p) for p in particles])
        
        T = T0
        cost = objective_function(best_particle)
        
        for _ in range(max_iter):
            for i in range(n_particles):
                # PSO update
                new_particle = particles[i] + np.random.uniform() * (best_particle - particles[i])
                
                # SA update
                new_cost = objective_function(new_particle)
                if acceptance_probability(cost, new_cost, T) > np.random.uniform():
                    particles[i] = new_particle
                    cost = new_cost
                    
                    if new_cost < objective_function(best_particle):
                        best_particle = new_particle
                        
                # Differential Evolution
                parents = particles[np.random.choice(range(n_particles), n_parents, replace=False)]
                donor_vector = parents[0] + F * (parents[1] - parents[2])
                crossover_mask = np.random.rand(self.dim) < CR
                trial_particle = np.where(crossover_mask, donor_vector, particles[i])
                
                trial_cost = objective_function(trial_particle)
                if trial_cost < new_cost:
                    particles[i] = trial_particle
                    cost = trial_cost
                    
                    if trial_cost < objective_function(best_particle):
                        best_particle = trial_particle
            
            T = alpha * T
            if T < T_min:
                break
        
        return best_particle