import numpy as np

class DynamicBirdSwarmOptimization:
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha=1.0, beta=0.5, inertia_min=0.4, inertia_max=0.9):
        self.budget = budget
        self.dim = dim
        self.num_birds = num_birds
        self.max_speed = max_speed
        self.alpha = alpha
        self.beta = beta
        self.inertia_min = inertia_min
        self.inertia_max = inertia_max

    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])]
        inertia_weight = self.inertia_max

        for _ in range(self.budget):
            for i in range(self.num_birds):
                velocities[i] = inertia_weight * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
        
        return best_position