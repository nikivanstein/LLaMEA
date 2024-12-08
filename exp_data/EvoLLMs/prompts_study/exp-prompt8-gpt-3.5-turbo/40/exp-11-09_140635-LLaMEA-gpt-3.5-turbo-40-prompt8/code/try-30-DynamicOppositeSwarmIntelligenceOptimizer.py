import numpy as np

class DynamicOppositeSwarmIntelligenceOptimizer:
    def __init__(self, budget, dim, swarm_size=20, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        global_best_position = best_position.copy()
        p_best_positions = swarm.copy()
        p_best_fitness = [func(p) for p in swarm]
        
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (p_best_positions[i] - swarm[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarm[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_component + social_component
                
                # Introducing mutation based on opposition-based learning
                if np.random.rand() < self.mutation_rate:
                    opposite_position = 2 * np.mean(swarm) - swarm[i]
                    swarm[i] = np.clip(opposite_position + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                else:
                    swarm[i] = np.clip(swarm[i] + velocities[i], -5.0, 5.0)
                
                current_fitness = func(swarm[i])
                if current_fitness < p_best_fitness[i]:
                    p_best_positions[i] = swarm[i]
                    p_best_fitness[i] = current_fitness
                    self.cognitive_weight = self.cognitive_weight * 0.95
                    self.social_weight = self.social_weight * 0.95
                if current_fitness < func(best_position):
                    best_position = swarm[i]
                if current_fitness < func(global_best_position):
                    global_best_position = swarm[i]
                    
        return global_best_position