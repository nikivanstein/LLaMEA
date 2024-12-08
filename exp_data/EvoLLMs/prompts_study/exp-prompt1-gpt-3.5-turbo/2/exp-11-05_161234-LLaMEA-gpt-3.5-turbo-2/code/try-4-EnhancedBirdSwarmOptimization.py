class EnhancedBirdSwarmOptimization(BirdSwarmOptimization):
    def __call__(self, func):
        birds = np.random.uniform(-5.0, 5.0, (self.num_birds, self.dim))
        velocities = np.zeros((self.num_birds, self.dim))
        best_position = birds[np.argmin([func(bird) for bird in birds])]
        
        for _ in range(self.budget):
            for i in range(self.num_birds):
                if np.random.uniform() < 0.1:  # Random restart with 10% probability
                    birds[i] = np.random.uniform(-5.0, 5.0, self.dim)
                    
                velocities[i] = self.alpha * velocities[i] + self.beta * np.random.uniform() * (best_position - birds[i])
                velocities[i] = np.clip(velocities[i], -self.max_speed, self.max_speed)
                
                birds[i] = np.clip(birds[i] + velocities[i], -5.0, 5.0)
                
                if func(birds[i]) < func(best_position):
                    best_position = birds[i]
                    
        return best_position