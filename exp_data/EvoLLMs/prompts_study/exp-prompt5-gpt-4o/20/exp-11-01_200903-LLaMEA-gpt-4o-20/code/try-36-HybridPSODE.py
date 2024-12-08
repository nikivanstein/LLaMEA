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
                    self.pbest[i] = self.particles[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.particles[i].copy()
            
            self.update_particles(func)

        return self.gbest

    def update_particles(self, func):
        w = 0.4 + 0.5 * (1 - self.fes / self.budget)  # Adjusted linear inertia weight
        c1, c2 = 1.5, 1.5
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive = c1 * r1 * (self.pbest[i] - self.particles[i])
            social = c2 * r2 * (self.gbest - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social
            self.particles[i] = self.particles[i] + self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

            if np.random.rand() < 0.3:  # Local search probability
                self.particles[i] = self.adaptive_mutation(func, self.particles[i])
                self.fes += 1

    def adaptive_mutation(self, func, target):
        F = np.random.uniform(0.4, 0.8)  # Adaptive mutation factor
        CR = 0.6  # Reduced crossover rate for diversity
        idxs = np.random.choice(self.swarm_size, 3, replace=False)
        a, b, c = self.particles[idxs]
        mutant = np.clip(a + F * (b - c), self.lb, self.ub)
        trial = np.copy(target)

        for j in range(self.dim):
            if np.random.rand() < CR:
                trial[j] = mutant[j]

        trial_score = func(trial)
        target_score = func(target)

        self.fes += 2
        return trial if trial_score < target_score else target