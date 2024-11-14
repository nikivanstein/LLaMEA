import numpy as np

class MultiSwarm_optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.swarm_size = 3
        self.max_iter = 1200
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.6
        self.temp = 15.0
        self.alpha = 0.9
        
    def __call__(self, func):
        def objective_function(x):
            return func(x)
        
        def accept_move(cost_diff, temp):
            return cost_diff < 0 or np.random.uniform(0, 1) < np.exp(-cost_diff / temp)
        
        # Initialize swarms
        swarms = [np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim)) for _ in range(self.swarm_size)]
        velocities = [np.zeros((self.pop_size, self.dim)) for _ in range(self.swarm_size)]
        personal_bests = [swarm.copy() for swarm in swarms]
        global_best = swarms[0][np.argmin([objective_function(p) for p in swarms[0]])
        
        for _ in range(self.max_iter):
            for j, swarm in enumerate(swarms):
                for i in range(self.pop_size):
                    # Update velocity
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[j][i] = self.inertia_weight * velocities[j][i] + self.c1 * r1 * (personal_bests[j][i] - swarm[i]) + self.c2 * r2 * (global_best - swarm[i])
                    # Update position
                    swarm[i] = np.clip(swarm[i] + velocities[j][i], -5.0, 5.0)
                    
                    # Simulated Annealing with Gaussian mutation
                    for _ in range(5):
                        new_particle = swarm[i] + np.random.normal(0, self.temp, size=self.dim)
                        cost_diff = objective_function(new_particle) - objective_function(swarm[i])
                        if accept_move(cost_diff, self.temp):
                            swarm[i] = new_particle
                        
                    # Update personal best
                    if objective_function(swarm[i]) < objective_function(personal_bests[j][i]):
                        personal_bests[j][i] = swarm[i].copy()
                    # Update global best
                    if objective_function(swarm[i]) < objective_function(global_best):
                        global_best = swarm[i].copy()
                
                self.temp *= self.alpha

        return global_best