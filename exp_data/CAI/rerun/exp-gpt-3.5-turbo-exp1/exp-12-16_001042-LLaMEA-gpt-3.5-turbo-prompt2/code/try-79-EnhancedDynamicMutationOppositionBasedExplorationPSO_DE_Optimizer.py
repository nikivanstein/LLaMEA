def levy_flight(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / abs(v) ** (1 / beta)
    return step

class EnhancedDynamicMutationOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        for _ in range(self.budget):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population) + levy_flight(self.dim)
            population += velocities
            # Remaining code stays the same
        return gbest