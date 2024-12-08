import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lb = -5.0
        self.ub = 5.0
        self.particles = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.swarm_size, self.dim))
        self.pbest = self.particles.copy()
        self.pbest_scores = np.full(self.swarm_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.fes = 0

    def __call__(self, func):
        while self.fes < self.budget:
            for i in range(self.swarm_size):
                if self.fes >= self.budget:
                    break
                score = func(self.particles[i])
                self.fes += 1
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.particles[i]
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.particles[i]
            
            self.update_particles(func)

        return self.gbest

    def update_particles(self, func):
        w = 0.5 * (1 - self.fes / self.budget)  # Adjusted inertia weight
        c1 = 2.0 * (1 - self.fes / self.budget)  # Dynamic learning factors
        c2 = 2.0 * self.fes / self.budget
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = c1 * r1 * (self.pbest[i] - self.particles[i])
            social = c2 * r2 * (self.gbest - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social
            self.particles[i] = self.particles[i] + self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            if self.fes < self.budget:
                self.local_search(func, i)

    def local_search(self, func, idx):
        # Local search around the best particle
        neighborhood = np.random.uniform(-0.1, 0.1, self.dim)
        candidate = self.particles[idx] + neighborhood
        candidate = np.clip(candidate, self.lb, self.ub)
        candidate_score = func(candidate)
        self.fes += 1
        if candidate_score < self.pbest_scores[idx]:
            self.pbest_scores[idx] = candidate_score
            self.pbest[idx] = candidate