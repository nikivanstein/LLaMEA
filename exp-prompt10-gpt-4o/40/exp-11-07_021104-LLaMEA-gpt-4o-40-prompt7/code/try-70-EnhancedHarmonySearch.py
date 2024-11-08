import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 8  # Slightly reduced size to focus on quality solutions
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_quality = np.full(self.harmony_memory_size, np.inf)
        self.memory_acceptance_rate = 0.9  # Increased for better exploration control
        self.adjustment_rate = 0.3  # Fine-tuned for consistent performance
        self.dynamic_bandwidth = 0.07  # More responsive bandwidth control

    def __call__(self, func):
        eval_count = 0
        initial_eval_limit = min(self.harmony_memory_size, self.budget)
        for i in range(initial_eval_limit):
            self.harmony_memory_quality[i] = func(self.harmony_memory[i])
            eval_count += 1

        while eval_count < self.budget:
            candidate_harmony = self._create_harmony()
            candidate_harmony = np.clip(candidate_harmony, self.lower_bound, self.upper_bound)
            candidate_value = func(candidate_harmony)
            eval_count += 1

            if candidate_value < np.max(self.harmony_memory_quality):
                worst_index = np.argmax(self.harmony_memory_quality)
                self.harmony_memory[worst_index] = candidate_harmony
                self.harmony_memory_quality[worst_index] = candidate_value

            self._dynamic_memory_adjustment(eval_count)

        return self.harmony_memory[np.argmin(self.harmony_memory_quality)]

    def _create_harmony(self):
        harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.memory_acceptance_rate:
                chosen_harmony = np.random.choice(self.harmony_memory_size)
                harmony[i] = self.harmony_memory[chosen_harmony, i]
                if np.random.rand() < self.adjustment_rate:
                    harmony[i] += (np.random.rand() - 0.5) * self.dynamic_bandwidth
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return harmony

    def _dynamic_memory_adjustment(self, eval_count):
        progress = eval_count / self.budget
        self.memory_acceptance_rate = 0.85 + 0.1 * np.cos(progress * np.pi)  # Adaptive rate change
        self.dynamic_bandwidth = 0.09 * (1 - progress)  # Dynamic bandwidth reduction