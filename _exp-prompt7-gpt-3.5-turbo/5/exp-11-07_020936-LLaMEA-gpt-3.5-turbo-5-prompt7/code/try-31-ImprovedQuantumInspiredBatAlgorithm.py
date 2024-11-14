import numpy as np

class ImprovedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.pop_size, self.loud, self.pulse, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        bats = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        vels = np.zeros((self.pop_size, self.dim))
        best_sol, best_fit = bats[0], func(bats[0])

        for _ in range(self.budget):
            for i in range(self.pop_size):
                if np.random.rand() > self.pulse:
                    freqs = np.clip(best_sol + self.alpha * (bats[i] - best_sol), -5.0, 5.0)
                    vels[i] += freqs * self.gamma
                else:
                    vels[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(vels[i])

                new_sol = np.clip(bats[i] + vels[i], -5.0, 5.0)
                new_fit = func(new_sol)

                if np.random.rand() < self.loud and new_fit < best_fit:
                    bats[i], best_sol, best_fit = new_sol, new_sol, new_fit

        return best_sol