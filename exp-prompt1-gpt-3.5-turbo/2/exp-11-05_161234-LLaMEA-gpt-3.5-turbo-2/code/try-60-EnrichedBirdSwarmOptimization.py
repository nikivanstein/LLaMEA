import numpy as np

class EnrichedBirdSwarmOptimization(BirdSwarmOptimization):
    def __call__(self, func):
        # Initialization
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])

        for _ in range(self.budget):
            for i in range(self.num_birds):
                # Update velocity
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)

                # Update position
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)

                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]

            # Introduce diversity
            if np.random.rand() < 0.05:
                birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))

        return best_position