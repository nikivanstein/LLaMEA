import numpy as np

class EnhancedHybridPSOSA:
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
        levy_flight_beta = 1.5
        
        def levy_flight():
            levy = np.zeros(self.dim)
            for d in range(self.dim):
                sigma = ((np.math.gamma(1 + levy_flight_beta) * np.math.sin(np.pi * levy_flight_beta / 2)) / (
                            np.math.gamma((1 + levy_flight_beta) / 2) * levy_flight_beta * (2 ** ((levy_flight_beta - 1) / 2)))) ** (1 / levy_flight_beta)
                u, v = np.random.normal(0, sigma), np.random.normal(0, sigma)
                step = u / (abs(v) ** (1 / levy_flight_beta))
                levy[d] = step
            return levy
        
        # Initialize particles
        particles = np.random.uniform(-5.0, 5.0, size=(n_particles, self.dim))
        best_particle = particles[np.argmin([objective_function(p) for p in particles])]
        
        T = T0
        cost = objective_function(best_particle)
        
        for _ in range(max_iter):
            for i in range(n_particles):
                # PSO update
                new_particle = particles[i] + np.random.uniform() * (best_particle - particles[i])
                
                # SA update
                new_particle += levy_flight()  # Levy flight step
                new_cost = objective_function(new_particle)
                
                if acceptance_probability(cost, new_cost, T) > np.random.uniform():
                    particles[i] = new_particle
                    cost = new_cost
                    
                    if new_cost < objective_function(best_particle):
                        best_particle = new_particle
            
            T = alpha * T
            if T < T_min:
                break
        
        return best_particle