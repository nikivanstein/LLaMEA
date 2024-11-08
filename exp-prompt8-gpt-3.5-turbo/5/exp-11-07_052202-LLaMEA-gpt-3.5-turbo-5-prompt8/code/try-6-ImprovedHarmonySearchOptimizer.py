import numpy as np

class ImprovedHarmonySearchOptimizer(HarmonySearchOptimizer):
    def __call__(self, func):
        def generate_new_harmony():
            new_harmony = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.pitch_adjust_rate:
                    index = np.random.randint(self.harmony_memory_size)
                    new_harmony[d] = self.harmony_memory[index, d]
            return new_harmony

        self.harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])

        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = generate_new_harmony()
            new_fitness = func(new_harmony)
            min_index = np.argmin(self.fitness_memory)
            if new_fitness < self.fitness_memory[min_index]:
                self.harmony_memory[min_index] = new_harmony
                self.fitness_memory[min_index] = new_fitness

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]