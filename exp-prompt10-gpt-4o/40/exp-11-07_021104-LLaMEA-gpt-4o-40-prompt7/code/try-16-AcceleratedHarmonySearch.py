import numpy as np
from concurrent.futures import ThreadPoolExecutor

class AcceleratedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_memory_size = 10  # Smaller initial memory for rapid convergence
        self.harmony_memory_rate_start = 0.9  # Higher initial exploration rate
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_memory_size, self.dim))
        self.harmony_memory_values = np.full(self.initial_memory_size, np.inf)
        self.pitch_adjustment_rate = 0.3  # Slightly reduced pitch rate
        self.bandwidth_start = 0.2  # Starting with a larger bandwidth

    def __call__(self, func):
        evaluations = 0
        self.harmony_memory_values[:self.initial_memory_size] = self._evaluate_parallel(self.harmony_memory[:self.initial_memory_size], func)
        evaluations += self.initial_memory_size

        while evaluations < self.budget:
            new_harmony = np.clip([self._generate_harmony(i) for i in range(self.dim)], self.lower_bound, self.upper_bound)
            new_value = func(new_harmony)
            evaluations += 1

            max_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[max_index]:
                self.harmony_memory[max_index] = new_harmony
                self.harmony_memory_values[max_index] = new_value

            if evaluations % (self.budget // 10) == 0:  # Dynamic memory size adjustment
                self._expand_memory(func)
            
            self._adaptive_adjustments(evaluations)

        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self, index):
        if np.random.rand() < self.harmony_memory_rate_start * (1 - index/self.dim):  # Dynamic harmony memory rate
            harmony_value = self.harmony_memory[np.random.randint(len(self.harmony_memory)), index]
            if np.random.rand() < self.pitch_adjustment_rate:
                harmony_value += self.bandwidth_start * (np.random.rand() - 0.5) * 2
        else:
            harmony_value = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony_value

    def _adaptive_adjustments(self, evaluations):
        progress = evaluations / self.budget
        self.bandwidth_start = 0.2 * (1 - progress**2)  # Quadratic reduction for bandwidth

    def _evaluate_parallel(self, harmonies, func):
        with ThreadPoolExecutor() as executor:
            results = executor.map(func, harmonies)
        return list(results)

    def _expand_memory(self, func):
        new_memory = np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))
        new_values = self._evaluate_parallel(new_memory, func)
        self.harmony_memory = np.vstack((self.harmony_memory, new_memory))
        self.harmony_memory_values = np.concatenate((self.harmony_memory_values, new_values))