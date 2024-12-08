import numpy as np

class Enhanced_DE_PSO_Optimizer(Dynamic_DE_PSO_Optimizer):
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99, levy_scale=0.1):
        super().__init__(budget, dim, npop, F, CR, w, c1, c2, F_decay, CR_decay, w_decay, c1_decay, c2_decay)
        self.levy_scale = levy_scale

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    def update_velocity(self, pbest, pop, velocity, gbest, w, c1, c2):
        for i in range(self.npop):
            velocity[i] = w * velocity[i] + c1 * np.random.rand() * (pbest[i] - pop[i]) + c2 * np.random.rand() * (gbest - pop[i]) + self.levy_scale * self.levy_flight(self.dim)
            velocity[i] = np.clip(velocity[i], -5.0, 5.0)
        return velocity