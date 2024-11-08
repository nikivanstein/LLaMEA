import numpy as np

class Improved_PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.initialize()

    def initialize(self):
        self.particles = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        self.velocities = np.zeros_like(self.particles)
        self.pbest_positions = self.particles.copy()
        self.gbest_position = self.particles[np.argmin([func(p) for p in self.particles])]
        self.best_value = func(self.gbest_position)
        self.T = 1.0
        self.current_position = np.mean(self.particles, axis=0)
        self.current_value = func(self.current_position)

    def __call__(self, func):
        def pso_step(w=0.5, c1=1.5, c2=1.5):
            nonlocal best_value
            for i in range(len(self.particles)):
                particle = self.particles[i]
                velocity = self.velocities[i]
                pbest_position = self.pbest_positions[i]

                r1, r2 = np.random.rand(), np.random.rand()
                new_velocity = w * velocity + c1 * r1 * (pbest_position - particle) + c2 * r2 * (self.gbest_position - particle)
                new_position = particle + new_velocity

                new_value = func(new_position)
                if new_value < self.best_value:
                    self.best_value = new_value
                    self.gbest_position = new_position

                if new_value < func(pbest_position):
                    self.pbest_positions[i] = new_position

                self.particles[i] = new_position
                self.velocities[i] = new_velocity

        def sa_step(alpha=0.95):
            nonlocal best_value
            new_position = self.current_position + np.random.normal(0, self.T, size=self.dim)
            new_position = np.clip(new_position, -5.0, 5.0)
            new_value = func(new_position)

            if new_value < self.current_value or np.random.rand() < np.exp((self.current_value - new_value) / self.T):
                self.current_position, self.current_value = new_position, new_value

            if new_value < self.best_value:
                self.best_value = new_value

        for _ in range(self.max_iter):
            pso_step()
            sa_step()
            self.T *= 0.95  # Cooling

        return self.best_value