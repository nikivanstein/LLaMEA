import numpy as np

class Enhanced_PSO_DIW:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.max_vel = 0.1
        self.c1 = 2.0
        self.c2 = 2.0
        self.particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.pbest_positions = self.particles.copy()
        self.pbest_values = np.full(self.swarm_size, np.inf)
        self.gbest_position = np.zeros(self.dim)
        self.gbest_value = np.inf
        self.inertia = 0.5

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

                if np.random.rand() < 0.35:
                    self.particles[i] += np.random.uniform(-0.1, 0.1, self.dim)  # Introduce random perturbation
            
            self.inertia = 0.5 + 0.5 * np.exp(-10.0 * _ / self.budget)  # Update inertia weight

        return self.gbest_value