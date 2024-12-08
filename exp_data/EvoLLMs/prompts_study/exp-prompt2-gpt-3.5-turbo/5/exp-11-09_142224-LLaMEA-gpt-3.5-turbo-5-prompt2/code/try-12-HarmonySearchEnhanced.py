import numpy as np

class HarmonySearchEnhanced(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(x) for x in harmony_memory])
        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = np.array([self.create_new_harmony(harmony_memory) for _ in range(self.dim)])
            new_fitness = func(new_harmony)
            worst_index = np.argmax(harmony_memory_fitness)
            if new_fitness < harmony_memory_fitness[worst_index]:
                harmony_memory[worst_index] = new_harmony
                harmony_memory_fitness[worst_index] = new_fitness
            if np.random.rand() < self.local_search_prob:
                local_search_harmony = self.local_search(harmony_memory[worst_index], func)
                local_search_fitness = func(local_search_harmony)
                if local_search_fitness < harmony_memory_fitness[worst_index]:
                    harmony_memory[worst_index] = local_search_harmony
                    harmony_memory_fitness[worst_index] = local_search_fitness
        return harmony_memory[np.argmin(harmony_memory_fitness)]
    
    def local_search(self, harmony, func):
        new_harmony = np.copy(harmony)
        for i in range(self.dim):
            new_harmony[i] += np.random.uniform(-self.bandwidth, self.bandwidth)
            new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
        return new_harmony