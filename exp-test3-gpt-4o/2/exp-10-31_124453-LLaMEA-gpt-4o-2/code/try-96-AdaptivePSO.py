import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = min(50, budget // dim)  # Adjust swarm size based on the budget and dimension
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evaluations = 0
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break
                # Evaluate fitness
                fitness = func(self.positions[i])
                self.evaluations += 1
                
                # Update personal best
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                
                # Update global best
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i]
            
            # Inertia weight adaptation with sinusoidal component
            w = 0.9 - (0.8 * np.cos((np.pi / 2) * (self.evaluations / self.budget))) + 0.1 * np.sin(2 * np.pi * (self.evaluations / self.budget))
            # Dynamic c1 and c2 with scaling factor
            c1 = 1.5 + 0.5 * (self.evaluations / self.budget) * 1.3  # Adjusted scaling factor to 1.3
            c2 = 2.0 - 0.5 * (self.evaluations / self.budget)  # As above, complementing c1
            
            # Update velocities and positions
            r1, r2 = np.random.random((2, self.swarm_size, self.dim))
            self.velocities = (w * self.velocities +
                               c1 * r1 * (self.personal_best_positions - self.positions) +
                               c2 * r2 * (self.global_best_position - self.positions))
            
            self.positions += self.velocities
            
            # Boundary handling
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
        
        # Return the best found solution
        return self.global_best_position, self.global_best_value