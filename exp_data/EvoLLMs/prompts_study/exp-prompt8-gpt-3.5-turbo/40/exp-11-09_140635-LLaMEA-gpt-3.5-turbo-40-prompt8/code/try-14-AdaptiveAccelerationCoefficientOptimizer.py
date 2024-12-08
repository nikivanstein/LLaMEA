import numpy as np

class AdaptiveAccelerationCoefficientOptimizer:
    def __init__(self, budget, dim, swarm_size=20, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        global_best_position = best_position.copy()
        p_best_positions = swarm.copy()
        
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                adaptive_acceleration_coefficient = np.abs(np.mean(swarm) - swarm[i]) / np.std(swarm)
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (p_best_positions[i] - swarm[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarm[i])
                velocities[i] = self.inertia_weight * velocities[i] + adaptive_acceleration_coefficient * (cognitive_component + social_component)
                swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                
                if func(swarm[i]) < func(best_position):
                    best_position = swarm[i]
                    p_best_positions[i] = swarm[i]
                if func(swarm[i]) < func(global_best_position):
                    global_best_position = swarm[i]
                    self.cognitive_weight = self.cognitive_weight * 0.9
                    self.social_weight = self.social_weight * 0.9
                    
        return global_best_position