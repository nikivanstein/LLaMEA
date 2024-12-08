import numpy as np

class DynamicSwarmSizeIntelligenceOptimizer:
    def __init__(self, budget, dim, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        swarm_size = 20
        swarm = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        global_best_position = best_position.copy()
        p_best_positions = swarm.copy()
        
        for _ in range(self.budget):
            for i in range(swarm_size):
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (p_best_positions[i] - swarm[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarm[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Introducing mutation based on opposition-based learning
                if np.random.rand() < self.mutation_rate:
                    opposite_position = 2 * np.mean(swarm) - swarm[i]
                    swarm[i] = np.clip(opposite_position + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                else:
                    swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                
                if func(swarm[i]) < func(best_position):
                    best_position = swarm[i]
                    p_best_positions[i] = swarm[i]
                if func(swarm[i]) < func(global_best_position):
                    global_best_position = swarm[i]
                    self.cognitive_weight = self.cognitive_weight * 0.9
                    self.social_weight = self.social_weight * 0.9
                    
            # Dynamic population resizing based on performance
            if np.random.rand() < 0.1:  # Adjust population size with 10% probability
                if np.random.rand() < 0.5 and swarm_size > 10:
                    swarm = swarm[:swarm_size//2]
                    velocities = velocities[:swarm_size//2]
                    p_best_positions = p_best_positions[:swarm_size//2]
                    swarm_size = swarm_size // 2
                elif swarm_size < 40:
                    new_swarm = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
                    new_velocities = np.zeros((swarm_size, self.dim))
                    new_p_best_positions = new_swarm.copy()
                    swarm = np.concatenate((swarm, new_swarm))
                    velocities = np.concatenate((velocities, new_velocities))
                    p_best_positions = np.concatenate((p_best_positions, new_p_best_positions))
                    swarm_size = swarm_size * 2
                
        return global_best_position