import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth
        self.lower_bound, self.upper_bound = -5.0, 5.0
        self.initialize_harmony_memory = lambda: np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.generate_new_harmony = lambda hm: np.clip(hm[np.random.randint(0, self.harmony_memory_size)] + np.where(np.random.rand(self.dim) < self.pitch_adjustment_rate, np.random.uniform(-self.bandwidth, self.bandwidth), 0), self.lower_bound, self.upper_bound)
        self.evaluate_harmony = lambda h: func(h)
        self.harmony_memory, self.harmony_scores = self.initialize_harmony_memory(), np.array([self.evaluate_harmony(h) for h in self.harmony_memory])

    def __call__(self, func):
        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony(self.harmony_memory)
            new_score = self.evaluate_harmony(new_harmony)
            min_index = np.argmin(self.harmony_scores)
            if new_score < self.harmony_scores[min_index]:
                self.harmony_memory[min_index], self.harmony_scores[min_index] = new_harmony, new_score
        return self.harmony_memory[np.argmin(self.harmony_scores)]