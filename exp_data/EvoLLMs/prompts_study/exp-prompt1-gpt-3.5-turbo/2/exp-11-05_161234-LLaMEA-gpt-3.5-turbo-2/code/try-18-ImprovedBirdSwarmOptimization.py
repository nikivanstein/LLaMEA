import numpy as np

class ImprovedBirdSwarmOptimization:
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, init_alpha=1.0, init_beta=0.5):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.max_speed = max_speed
        self.init_alpha = init_alpha
        self.init_beta = init_beta

    def __call__(self, func):
        # Initialization
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])]
        
        alpha = self.init_alpha
        beta = self.init_beta
        
        for _ in range(self.budget):
            for i in range(self.num_birds):
                # Update velocity
                velocities[i] = alpha * velocities[i] + beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                
                # Update position
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                
                # Update best position
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
            
            # Update alpha and beta dynamically
            alpha = max(0.9 * alpha, 0.1)
            beta = min(1.1 * beta, 0.9)
                    
        return best_position