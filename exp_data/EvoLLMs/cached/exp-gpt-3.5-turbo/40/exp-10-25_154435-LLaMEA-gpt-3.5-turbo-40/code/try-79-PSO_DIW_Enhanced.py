import numpy as np

class PSO_DIW_Enhanced(PSO_DIW):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.max_vel = 0.2
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                fitness = func(self.particles[i])
                if fitness < self.pbest_values[i]:
                    self.pbest_values[i] = fitness
                    self.pbest_positions[i] = self.particles[i].copy()
                
                if fitness < self.gbest_value:
                    self.gbest_value = fitness
                    self.gbest_position = self.particles[i].copy()

                r1, r2 = np.random.uniform(0, 1, 2)
                self.velocities[i] = self.inertia * self.velocities[i] + self.c1 * r1 * (self.pbest_positions[i] - self.particles[i]) + self.c2 * r2 * (self.gbest_position - self.particles[i])
                np.clip(self.velocities[i], -self.max_vel, self.max_vel, out=self.velocities[i])
                self.particles[i] += self.velocities[i]
            
            self.inertia = 0.5 + 0.5 * np.exp(-10.0 * _ / self.budget)  # Update inertia weight
            self.c1 = 2.0 + 0.8 * np.exp(-6.0 * _ / self.budget)  # Update cognitive parameter
            self.c2 = 2.0 - 1.0 * np.exp(-8.0 * _ / self.budget)  # Update social parameter
        
        return self.gbest_value