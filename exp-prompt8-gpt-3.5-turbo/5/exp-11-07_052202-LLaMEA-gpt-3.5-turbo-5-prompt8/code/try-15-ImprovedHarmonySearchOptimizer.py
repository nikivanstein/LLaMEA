import numpy as np

class ImprovedHarmonySearchOptimizer:
    def __init__(self, budget, dim, harmony_memory_size=10, pitch_adjust_rate=0.3, pitch_bandwidth=0.01):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_bandwidth = pitch_bandwidth

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_harmony_memory(new_solution, new_fitness):
            min_index = np.argmin(self.fitness_memory)
            if new_fitness < self.fitness_memory[min_index]:
                self.harmony_memory[min_index] = new_solution
                self.fitness_memory[min_index] = new_fitness

        def generate_new_harmony():
            new_harmony = np.zeros(self.dim)
            indices = np.random.rand(self.dim) < self.pitch_adjust_rate
            selected_indices = np.random.randint(0, self.harmony_memory_size, size=sum(indices))
            new_harmony[indices] = self.harmony_memory[selected_indices, indices]
            new_harmony[~indices] = np.random.uniform(-self.pitch_bandwidth, self.pitch_bandwidth, size=sum(~indices))
            return new_harmony

        self.harmony_memory = initialize_harmony_memory()
        self.fitness_memory = np.array([func(harmony) for harmony in self.harmony_memory])

        for _ in range(self.budget - self.harmony_memory_size):
            new_harmony = generate_new_harmony()
            new_fitness = func(new_harmony)
            update_harmony_memory(new_harmony, new_fitness)

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index]

# Usage:
budget = 1000
dim = 10
optimizer = ImprovedHarmonySearchOptimizer(budget, dim)
optimized_solution = optimizer(lambda x: np.sum(x ** 2))