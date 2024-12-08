import numpy as np

class AdaptiveSwarmLearningRateOptimizer:
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, initial_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        best_positions = [swarms[i][np.argmin([func(p) for p in swarms[i]])] for i in range(self.num_swarms)]
        global_best_position = best_positions[0].copy()
        learning_rates = np.full((self.num_swarms, self.swarm_size), self.initial_learning_rate)

        for _ in range(self.budget):
            for i in range(self.num_swarms):
                for j in range(self.swarm_size):
                    cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (best_positions[i] - swarms[i][j])
                    social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarms[i][j])
                    adaptive_inertia = self.inertia_weight * (1 - _ / self.budget)

                    velocities[i][j] = adaptive_inertia * velocities[i][j] + cognitive_component + social_component

                    if np.random.rand() < learning_rates[i][j]:
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
                        learning_rates[i][j] *= 0.9  # Adaptive learning rate mechanism

        return global_best_position