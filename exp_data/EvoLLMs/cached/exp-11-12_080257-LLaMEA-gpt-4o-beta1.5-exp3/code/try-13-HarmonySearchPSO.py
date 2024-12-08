import numpy as np

class HarmonySearchPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = max(5, 2 * dim)
        self.harmony_consideration_rate = 0.95
        self.random_selection_rate = 0.05
        self.adjustment_rate = 0.3
        self.eval_count = 0
        
    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def evaluate_harmony_memory(self, memory, func):
        fitness = np.array([func(ind) for ind in memory])
        self.eval_count += len(memory)
        return fitness

    def new_harmony(self, memory):
        harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.harmony_consideration_rate:
                harmony[i] = memory[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.adjustment_rate:
                    harmony[i] += np.random.uniform(-0.1, 0.1)
            else:
                harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        harmony = np.clip(harmony, self.lower_bound, self.upper_bound)
        return harmony

    def particle_swarm_optimization(self, memory, func):
        velocity = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim))
        personal_best = memory.copy()
        personal_best_fitness = self.evaluate_harmony_memory(personal_best, func)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]

        while self.eval_count < self.budget:
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = 0.5 * velocity + r1 * (personal_best - memory) + r2 * (global_best - memory)
            memory = np.clip(memory + velocity, self.lower_bound, self.upper_bound)
            fitness = self.evaluate_harmony_memory(memory, func)

            for i in range(self.harmony_memory_size):
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = memory[i]
                    personal_best_fitness[i] = fitness[i]

            global_best_idx = np.argmin(personal_best_fitness)
            global_best = personal_best[global_best_idx]

            if self.eval_count >= self.budget:
                break
        
        return global_best

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        harmony_fitness = self.evaluate_harmony_memory(harmony_memory, func)

        while self.eval_count < self.budget:
            if self.eval_count >= self.budget:
                break
            new_harmony = self.new_harmony(harmony_memory)
            new_harmony_fitness = func(new_harmony)
            self.eval_count += 1

            worst_idx = np.argmax(harmony_fitness)
            if new_harmony_fitness < harmony_fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_fitness[worst_idx] = new_harmony_fitness

        best_harmony = self.particle_swarm_optimization(harmony_memory, func)
        return best_harmony