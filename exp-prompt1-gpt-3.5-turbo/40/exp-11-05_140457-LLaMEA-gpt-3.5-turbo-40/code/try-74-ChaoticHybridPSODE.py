import numpy as np

class ChaoticHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.chaos_prob = 0.1
        self.chaos_intensity = 0.2

    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.chaos_prob:  # Chaotic Perturbation
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.chaos_intensity, self.chaos_intensity, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            else:
                super().__call__(func)
        return self.global_best