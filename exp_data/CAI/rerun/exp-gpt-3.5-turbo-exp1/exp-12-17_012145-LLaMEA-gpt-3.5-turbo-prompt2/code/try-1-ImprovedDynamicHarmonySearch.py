import numpy as np

class ImprovedDynamicHarmonySearch(DynamicHarmonySearch):
    def __init__(self, budget, dim, levy_step_size=0.1):
        super().__init__(budget, dim)
        self.levy_step_size = levy_step_size

    def levy_flight_step(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1/beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v)**(1/beta)
        return step

    def improvise_harmony(self, harmony_memory, par, bandwidth):
        new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
        for i in range(self.dim):
            if np.random.uniform() < par:
                if np.random.rand() < 0.1:
                    new_harmony[i] += self.levy_step_size * self.levy_flight_step()
                else:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
        return new_harmony