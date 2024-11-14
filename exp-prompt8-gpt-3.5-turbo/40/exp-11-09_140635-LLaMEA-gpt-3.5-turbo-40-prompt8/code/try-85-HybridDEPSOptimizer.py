class HybridDEPSOptimizer:
    def __init__(self, budget, dim, num_particles=10, f=0.5, cr=0.9, w=0.7):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f = f
        self.cr = cr
        self.w = w

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])
        velocity = np.zeros((self.num_particles, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                trial_vector = np.clip(swarm[i] + self.f * (swarm[np.random.choice(self.num_particles)] - swarm[np.random.choice(self.num_particles)]), -5.0, 5.0)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == np.random.choice(self.dim):
                        swarm[i, j] = trial_vector[j]
                velocity[i] = self.w * velocity[i] + np.random.rand() * (best_position - swarm[i])
                swarm[i] = np.clip(swarm[i] + velocity[i], -5.0, 5.0)
                if func(swarm[i]) < func(best_position):
                    best_position = swarm[i]
        return best_position