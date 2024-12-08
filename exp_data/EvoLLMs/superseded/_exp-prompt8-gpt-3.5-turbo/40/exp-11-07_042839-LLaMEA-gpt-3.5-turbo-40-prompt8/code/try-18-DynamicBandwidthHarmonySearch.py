import numpy as np

class DynamicBandwidthHarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01, elite_percentage=0.1):
        self.budget, self.dim, self.harmony_memory_size, self.pitch_adjustment_rate, self.bandwidth, self.elite_percentage = budget, dim, harmony_memory_size, pitch_adjustment_rate, bandwidth, elite_percentage

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

        def generate_new_harmony(harmony_memory, elite_count):
            random_index = np.random.randint(0, self.harmony_memory_size if elite_count == 0 else elite_count)
            new_harmony = np.clip(np.copy(harmony_memory[random_index] + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim)), -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        harmony_scores = np.array([func(h) for h in harmony_memory])
        elite_count = int(self.elite_percentage * self.harmony_memory_size)
        for _ in range(self.budget):
            new_harmony = generate_new_harmony(harmony_memory, elite_count)
            new_score = func(new_harmony)
            min_index = np.argmin(harmony_scores)
            if new_score < harmony_scores[min_index]:
                harmony_memory[min_index], harmony_scores[min_index] = new_harmony, new_score
                elite_count = int(self.elite_percentage * self.harmony_memory_size)
        return harmony_memory[np.argmin(harmony_scores)]