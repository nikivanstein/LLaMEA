import numpy as np

class ImprovedBirdSwarmOptimization(BirdSwarmOptimization):
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha_min=0.5, beta_min=0.3, alpha_max=1.5, beta_max=0.7):
        super().__init__(budget, dim, num_birds, max_speed)
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.alpha_max = alpha_max
        self.beta_max = beta_max

    def __call__(self, func):
        # Initialization
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])
        
        for _ in range(self.budget):
            for i in range(self.num_birds):
                # Update velocity
                self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (self.budget - _) / self.budget
                self.beta = self.beta_min + (self.beta_max - self.beta_min) * (self.budget - _) / self.budget
                
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                
                # Update position
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                
                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
        return best_position