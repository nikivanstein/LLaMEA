import numpy as np

class EnhancedGradientGuidedMultiSwarmDynamicOppositeOptimizer:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, gradient_weight=0.5, initial_mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.gradient_weight = gradient_weight
        self.initial_mutation_rate = initial_mutation_rate

    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        best_positions = [swarms[i][np.argmin([func(p) for p in swarms[i]])] for i in range(self.num_swarms)]
        global_best_position = best_positions[0].copy()
        mutation_rate = self.initial_mutation_rate

        for _ in range(self.budget):
            gradients = [np.gradient([func(p) for p in swarms[i]]) for i in range(self.num_swarms)]
            for i in range(self.num_swarms):
                for j in range(self.swarm_size):
                    cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (best_positions[i] - swarms[i][j])
                    social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarms[i][j])
                    gradient_component = self.gradient_weight * gradients[i][j]
                    adaptive_inertia = self.inertia_weight * (1 - _ / self.budget)
                    velocities[i][j] = adaptive_inertia * velocities[i][j] + cognitive_component + social_component + gradient_component
                    
                    if np.random.rand() < mutation_rate:
                        opposite_position = 2 * np.mean(swarms[i]) - swarms[i][j]
                        swarms[i][j] = np.clip(opposite_position + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                    else:
                        swarms[i][j] = np.clip(swarms[i][j] + velocities[i][j], -5.0, 5.0)
                    
                    if func(swarms[i][j]) < func(best_positions[i]):
                        best_positions[i] = swarms[i][j]
                    if func(swarms[i][j]) < func(global_best_position):
                        global_best_position = swarms[i][j]
                        self.cognitive_weight = self.cognitive_weight * 0.9
                        self.social_weight = self.social_weight * 0.9
                        mutation_rate *= 0.95
                    
        return global_best_position