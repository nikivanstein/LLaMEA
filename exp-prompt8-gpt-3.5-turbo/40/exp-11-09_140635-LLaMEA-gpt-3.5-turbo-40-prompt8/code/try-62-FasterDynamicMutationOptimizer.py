import numpy as np

class FasterDynamicMutationOptimizer:
    def __init__(self, budget, dim, swarm_size=20, num_swarms=5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = num_swarms
        self.mutation_rates = np.full((self.num_swarms, self.swarm_size), np.random.uniform(0, 1))
    
    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        best_positions = [swarms[i][np.argmin([func(p) for p in swarms[i]])] for i in range(self.num_swarms)]
        global_best_position = best_positions[0].copy()
        swarm_scores = [func(p) for p in best_positions]
        
        for _ in range(self.budget):
            for i in range(self.num_swarms):
                for j in range(self.swarm_size):
                    cognitive_component = np.random.rand(self.dim) * (best_positions[i] - swarms[i][j])
                    social_component = np.random.rand(self.dim) * (global_best_position - swarms[i][j])
                    adaptive_inertia = 0.9 + 0.1 * _ / self.budget
                    
                    velocities[i][j] = adaptive_inertia * velocities[i][j] + cognitive_component + social_component
                    
                    if np.random.rand() < self.mutation_rates[i][j]:
                        opposite_position = 2 * np.mean(swarms[i]) - swarms[i][j]
                        swarms[i][j] = np.clip(opposite_position + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                    else:
                        swarms[i][j] = np.clip(swarms[i][j] + velocities[i][j], -5.0, 5.0)
                    
                    current_score = func(swarms[i][j])
                    if current_score < func(best_positions[i]):
                        best_positions[i] = swarms[i][j]
                        swarm_scores[i] = current_score
                    if current_score < func(global_best_position):
                        global_best_position = swarms[i][j]
                        self.mutation_rates[i][j] *= 0.9 + 0.1 * swarm_scores[i] / current_score
        return global_best_position