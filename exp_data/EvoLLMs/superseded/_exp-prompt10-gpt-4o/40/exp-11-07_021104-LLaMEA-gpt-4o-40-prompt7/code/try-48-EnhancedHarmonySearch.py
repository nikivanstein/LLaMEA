import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Reduced size for faster adaptation
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_values = np.array([np.inf] * self.harmony_memory_size)
        self.harmony_memory_rate = 0.9  # Enhanced for better exploration
        self.pitch_adjustment_rate = 0.3  # Adjusted for stability and exploration
        self.bandwidth = 0.1  # Fixed bandwidth for consistent pitch adjustment

    def __call__(self, func):
        evaluations = 0
        initial_evals = min(self.harmony_memory_size, self.budget)
        self.harmony_memory_values[:initial_evals] = [func(hm) for hm in self.harmony_memory[:initial_evals]]
        evaluations += initial_evals
        
        while evaluations < self.budget:
            new_harmony = self._generate_harmony()
            new_value = func(new_harmony)
            evaluations += 1

            worst_index = np.argmax(self.harmony_memory_values)
            if new_value < self.harmony_memory_values[worst_index]:
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_values[worst_index] = new_value

            self._dynamic_tuning(evaluations)
        
        return self.harmony_memory[np.argmin(self.harmony_memory_values)]

    def _generate_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_memory_rate:
                selected_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size), i]
                harmony[i] = selected_harmony + np.random.uniform(-self.bandwidth, self.bandwidth) if np.random.rand() < self.pitch_adjustment_rate else selected_harmony
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return np.clip(harmony, self.lower_bound, self.upper_bound)

    def _dynamic_tuning(self, evaluations):
        progress = evaluations / self.budget
        self.harmony_memory_rate = 0.9 - 0.2 * progress  # Decreasing rate for refined search