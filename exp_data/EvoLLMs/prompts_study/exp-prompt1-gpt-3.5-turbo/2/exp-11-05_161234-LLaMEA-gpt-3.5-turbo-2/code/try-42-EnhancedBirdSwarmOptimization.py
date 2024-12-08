import numpy as np

class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__(budget, dim, num_birds, max_speed, alpha, beta)
        self.gamma = gamma

    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        prev_best_position = best_position.copy()  # Store previous best position

        for _ in range(self.budget):
            for i in range(self.num_birds):
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)

                # Introducing diversity-based exploration
                if np.random.uniform() < self.gamma:
                    birds[i] = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)

                if func(birds[i]) < func(best_position):
                    best_position = birds[i]

            # Update gamma to decrease exploration as optimization progresses
            self.gamma *= 0.99

            # Reinitialize if best position doesn't change
            if np.array_equal(best_position, prev_best_position):
                birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
                prev_best_position = best_position.copy()

        return best_position