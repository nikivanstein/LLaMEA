import numpy as np

class EnhancedDynamicBirdSwarmOptimization(DynamicBirdSwarmOptimization):
    def __init__(self, budget, dim, num_birds=20, max_speed=0.1, alpha=1.0, beta=0.5, inertia_min=0.4, inertia_max=0.9, levy_prob=0.05):
        super().__init__(budget, dim, num_birds, max_speed, alpha, beta, inertia_min, inertia_max)
        self.levy_prob = levy_prob

    def levy_flight(self, position):
        levy_alpha = 1.5
        levy_beta = 0.5
        sigma = ((math.gamma(1 + levy_alpha) * np.sin(np.pi * levy_alpha / 2)) / (math.gamma((1 + levy_alpha) / 2) * levy_alpha * 2 ** ((levy_alpha - 1) / 2))) ** (1 / levy_alpha)
        u = np.random.normal(0, sigma ** 2)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / levy_alpha)
        position += step
        return np.clip(position, -5.0, 5.0)

    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])]
        inertia_weight = self.inertia_max

        for _ in range(self.budget):
            for i in range(self.num_birds):
                if np.random.rand() < self.levy_prob:
                    birds[i] = self.levy_flight(birds[i])
                else:
                    velocities[i] = inertia_weight * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                    velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                    birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                    if func(birds[i]) < func(best_position):
                        best_position = birds[i]
                    
            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
        
        return best_position