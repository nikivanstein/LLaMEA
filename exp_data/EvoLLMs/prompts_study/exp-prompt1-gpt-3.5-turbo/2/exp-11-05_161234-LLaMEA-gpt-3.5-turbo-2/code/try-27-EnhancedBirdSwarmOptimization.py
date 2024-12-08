import numpy as np

class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __call__(self, func):
        # Initialization
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        inertia_weight = 0.5

        for _ in range(self.budget):
            for i in range(self.num_birds):
                # Update velocity with dynamic inertia weight
                velocities[i] = inertia_weight * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                
                # Update position
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                
                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
            
            inertia_weight = 0.5 + 0.5 * (self.budget - _) / self.budget # Update inertia weight

        return best_position