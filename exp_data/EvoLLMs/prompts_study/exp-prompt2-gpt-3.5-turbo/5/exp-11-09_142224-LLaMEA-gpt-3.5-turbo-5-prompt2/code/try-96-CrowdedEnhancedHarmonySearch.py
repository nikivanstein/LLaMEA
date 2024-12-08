import numpy as np

class CrowdedEnhancedHarmonySearch(EnhancedHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.dim,))
        harmony_memory_fitness = func(harmony_memory)
        for _ in range(self.budget - 1):
            self.bandwidth = 0.01 * (self.upper_bound - self.lower_bound) * (1.0 - _ / self.budget)  # Dynamic bandwidth adaptation
            new_harmony = self.create_new_harmony(harmony_memory)
            new_fitness = func(new_harmony)
            if new_fitness < harmony_memory_fitness:
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
            crowding_distances = self.calculate_crowding_distances([harmony_memory, new_harmony])  # Calculate crowding distances
            if crowding_distances[0] < crowding_distances[1]:  # Update harmony_memory based on better crowding distance
                harmony_memory = new_harmony
                harmony_memory_fitness = new_fitness
        return harmony_memory

    def calculate_crowding_distances(self, harmonies):
        crowding_distances = []
        for i, harmony in enumerate(harmonies):
            distances = [np.linalg.norm(harmony - other) for j, other in enumerate(harmonies) if i != j]
            crowding_distances.append(sum(distances))
        return crowding_distances