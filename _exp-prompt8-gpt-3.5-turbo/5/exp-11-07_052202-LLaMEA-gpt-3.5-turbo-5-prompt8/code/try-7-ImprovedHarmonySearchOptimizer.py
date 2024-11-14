import numpy as np

class ImprovedHarmonySearchOptimizer(HarmonySearchOptimizer):
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        super().__init__(budget, dim, harmony_memory_size, pitch_adjust_rate, pitch_bandwidth)

    def __call__(self, func):
        def generate_new_harmony():
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    index = np.random.randint(self.harmony_memory_size)
                    new_harmony[d] = self.harmony_memory[index, d]
                else:
                    new_harmony[d] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth)
            return new_harmony

        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = generate_new_harmony()
            new_fitness = func(new_harmony)
            min_index = np.argmin(self.fitness_memory)
            if new_fitness < self.fitness_memory[min_index]:
                self.harmony_memory[min_index] = new_harmony
                self.fitness_memory[min_index] = new_fitness

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]