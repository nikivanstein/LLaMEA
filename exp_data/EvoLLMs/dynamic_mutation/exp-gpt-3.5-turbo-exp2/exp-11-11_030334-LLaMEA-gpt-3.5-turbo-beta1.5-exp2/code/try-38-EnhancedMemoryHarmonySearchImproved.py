import numpy as np

class EnhancedMemoryHarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.bandwidth = 0.1
        self.initial_bandwidth = 0.1
        self.harmony_memory_size = 20
        self.hm_accept_rate = 0.95
        self.pitch_adjust_rate = 0.7
        self.local_search_rate = 0.3
        self.memory_strength = 0.5
        self.memory_update_rate = 0.1  # New line

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

        def adjust_value_to_bounds(value):
            return max(self.lower_bound, min(self.upper_bound, value))

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = float('inf')
        fitness_history = []
        memory_strength_history = []  # New line

        for _ in range(self.budget):
            new_solution = np.clip(harmony_memory[np.random.randint(self.harmony_memory_size)] + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim), self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)

            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

            if np.random.rand() < self.hm_accept_rate:
                replace_index = np.random.randint(self.harmony_memory_size)
                if new_fitness < func(harmony_memory[replace_index]):
                    harmony_memory[replace_index] = new_solution

            if np.random.rand() < self.local_search_rate:
                local_search_space = np.clip(new_solution + np.random.uniform(-self.bandwidth, self.bandwidth, self.dim), self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_space)
                if local_fitness < new_fitness:
                    new_solution = local_search_space
                    new_fitness = local_fitness

            fitness_history.append(new_fitness)
            if len(fitness_history) > 1 and fitness_history[-1] < fitness_history[-2]:
                self.bandwidth *= self.pitch_adjust_rate
            else:
                self.bandwidth = self.initial_bandwidth

            self.bandwidth = adjust_value_to_bounds(self.bandwidth)

            if len(fitness_history) > 2 and fitness_history[-1] < fitness_history[-2] and fitness_history[-2] < fitness_history[-3]:
                self.bandwidth *= self.memory_strength
                memory_strength_history.append(self.memory_strength)  # New block
                if len(memory_strength_history) > 1 and memory_strength_history[-1] < memory_strength_history[-2]:
                    self.memory_strength *= 1.1
                else:
                    self.memory_strength /= 1.1

            if (len(fitness_history) > 1) and (fitness_history[-1] < fitness_history[-2]):
                self.memory_strength += self.memory_update_rate * (1 - self.memory_strength)

        return best_solution