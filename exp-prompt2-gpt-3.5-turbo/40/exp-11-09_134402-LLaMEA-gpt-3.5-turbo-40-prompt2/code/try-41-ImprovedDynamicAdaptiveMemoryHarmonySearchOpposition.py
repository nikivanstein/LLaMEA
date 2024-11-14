import numpy as np

class ImprovedDynamicAdaptiveMemoryHarmonySearchOpposition:
    def __init__(self, budget, dim, harmony_memory_size=10, initial_bandwidth=0.01, bandwidth_range=(0.01, 0.1), initial_pitch_adjustment_rate=0.2, pitch_adjustment_range=(0.1, 0.5), memory_consideration_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.initial_bandwidth = initial_bandwidth
        self.bandwidth_range = bandwidth_range
        self.initial_pitch_adjustment_rate = initial_pitch_adjustment_rate
        self.pitch_adjustment_range = pitch_adjustment_range
        self.memory_consideration_prob = memory_consideration_prob
    
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        
        def update_harmony_memory(harmony_memory, new_solution):
            harmony_memory = np.vstack((harmony_memory, new_solution))
            harmony_memory = harmony_memory[np.argsort(func(harmony_memory))]
            return harmony_memory[:self.harmony_memory_size]
        
        def improvise(harmony_memory, bandwidth, pitch_adjustment_rate):
            new_solution = np.copy(harmony_memory[np.random.randint(self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < pitch_adjustment_rate:
                    pitch_range = np.random.uniform(*self.pitch_adjustment_range)
                    new_solution[i] += np.random.uniform(-pitch_range, pitch_range)
                    new_solution[i] = np.clip(new_solution[i], -5.0, 5.0)
                if np.random.rand() < self.memory_consideration_prob:
                    new_solution[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                new_solution[i] = 2 * np.mean(harmony_memory[:, i]) - new_solution[i]  # Opposition-based learning
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        bandwidth = self.initial_bandwidth
        pitch_adjustment_rate = self.initial_pitch_adjustment_rate
        for _ in range(self.budget):
            bandwidth = np.clip(bandwidth + np.random.uniform(-0.005, 0.005), *self.bandwidth_range)  # Adaptive bandwidth adjustment
            pitch_adjustment_rate = np.clip(pitch_adjustment_rate + np.random.uniform(-0.025, 0.025), *self.pitch_adjustment_range)  # Adaptive pitch adjustment rate
            new_solution = improvise(harmony_memory, bandwidth, pitch_adjustment_rate)
            if func(new_solution) < func(harmony_memory[-1]):
                harmony_memory = update_harmony_memory(harmony_memory, new_solution)
        
        return harmony_memory[0]