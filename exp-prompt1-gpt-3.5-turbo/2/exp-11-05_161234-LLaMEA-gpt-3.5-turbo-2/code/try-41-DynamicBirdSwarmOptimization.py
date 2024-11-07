import numpy as np

class DynamicBirdSwarmOptimization:
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, initial_alpha=1.0, initial_beta=0.5):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.max_speed = max_speed
        self.alpha = initial_alpha
        self.beta = initial_beta

    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        
        for _ in range(self.budget):
            improvement_rate = sum(func(birds[i]) < func(best_position) for i in range(self.num_birds)) / self.num_birds
            self.alpha = 1.0 - improvement_rate
            self.beta = 0.5 + improvement_rate
            
            for i in range(self.num_birds):
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
        return best_position