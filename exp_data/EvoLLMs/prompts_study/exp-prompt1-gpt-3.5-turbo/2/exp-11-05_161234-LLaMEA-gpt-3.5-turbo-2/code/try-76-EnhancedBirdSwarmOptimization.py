import numpy as np

class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha=1.0, beta=0.5, inertia_weight=0.5):
        super().__init__(budget, dim, num_birds, max_speed, alpha, beta)
        self.inertia_weight = inertia_weight

    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])

        for _ in range(self.budget):
            for i in range(self.num_birds):
                weights = np.random.uniform(-1.0, 1.0, self.dim)
                opp_position = 2.0 * best_position - birds[i]
                velocities[i] = self.inertia_weight * velocities[i] + self.alpha * weights * (best_position - birds[i]) + self.beta * weights * (opp_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)

                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)

                if func(birds[i]) < func(best_position):
                    best_position = birds[i]

        return best_position