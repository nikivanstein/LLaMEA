import numpy as np

class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha=1.0, beta=0.5, mutation_rate=0.05):
        super().__init__(budget, dim, num_birds, max_speed, alpha, beta)
        self.mutation_rate = mutation_rate

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
                
                # Mutation
                if np.random.uniform() < self.mutation_rate:
                    birds[i] = np.random.uniform(-5.0, 5.0, self.dim)
                
                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
        return best_position