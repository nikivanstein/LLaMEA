import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        n_particles = 20
        max_iter = self.budget // n_particles
        alpha = 0.9
        T0 = 1.0
        T_min = 1e-5
        inertia_max = 0.9
        inertia_min = 0.4
        
        def acceptance_probability(cost, new_cost, T):
            if new_cost < cost:
                return 1.0
            return np.exp((cost - new_cost) / T)
        
        # Initialize particles
        particles = np.random.uniform(-5.0, 5.0, size=(n_particles, self.dim))
        best_particle = particles[np.argmin([objective_function(p) for p in particles])
        
        T = T0
        cost = objective_function(best_particle)
        inertia_weight = inertia_max
        
        for _ in range(max_iter):
            for i in range(n_particles):
                # PSO update with dynamic inertia weight
                new_particle = particles[i] + inertia_weight * np.random.uniform() * (best_particle - particles[i])
                
                # SA update
                new_cost = objective_function(new_particle)
                if acceptance_probability(cost, new_cost, T) > np.random.uniform():
                    particles[i] = new_particle
                    cost = new_cost
                    
                    if new_cost < objective_function(best_particle):
                        best_particle = new_particle
            
            T = alpha * T
            inertia_weight = max(inertia_max - (_ / max_iter) * (inertia_max - inertia_min), inertia_min)
            
            if T < T_min:
                break
        
        return best_particle