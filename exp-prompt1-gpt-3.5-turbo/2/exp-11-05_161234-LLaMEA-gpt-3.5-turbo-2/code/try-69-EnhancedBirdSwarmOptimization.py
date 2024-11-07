import numpy as np

class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        global_best_position = best_position  # Initialize global best
        
        for _ in range(self.budget):
            for i in range(self.num_birds):
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * \
                                (global_best_position - birds[i])  # Update velocity with global best
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                if func(birds[i]) < func(global_best_position):  # Update global best
                    global_best_position = birds[i]
        
        return best_position