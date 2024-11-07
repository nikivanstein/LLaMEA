import numpy as np

class ImprovedBirdSwarmOptimization(BirdSwarmOptimization):
    def __call__(self, func):
        # Initialization
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        inertia_weight = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, self.budget))  # Dynamic inertia weight

        for t in range(self.budget):
            for i in range(self.num_birds):
                # Update velocity
                velocities[i] = inertia_weight[t] * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                
                # Update position
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                
                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
        return best_position