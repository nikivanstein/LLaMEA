import numpy as np

class DynamicAdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hm_size = 20  
        self.hmcr = 0.85    
        self.par = 0.35     
        self.bw = 0.05      
        self.mutation_prob = 0.1

    def initialize_harmony_memory(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

    def evaluate_harmonies(self, harmonies, func):
        return np.array([func(harmony) for harmony in harmonies])

    def update_parameters(self, iteration, max_iterations):
        self.hmcr = 0.85 - 0.35 * (iteration / max_iterations)
        self.par = 0.35 + 0.25 * (iteration / max_iterations)
        self.bw = 0.05 * (1 - iteration / max_iterations)

    def mutate_harmony(self, harmony):
        if np.random.rand() < self.mutation_prob:
            mutation_index = np.random.randint(self.dim)
            harmony[mutation_index] += np.random.normal(0, self.bw)
            harmony[mutation_index] = np.clip(harmony[mutation_index], self.lower_bound, self.upper_bound)
        return harmony

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        harmony_values = self.evaluate_harmonies(harmony_memory, func)
        evaluations = self.hm_size
        best_harmony = harmony_memory[np.argmin(harmony_values)]
        max_iterations = self.budget // self.hm_size

        while evaluations < self.budget:
            for iteration in range(max_iterations):
                self.update_parameters(iteration, max_iterations)
                new_harmony = np.copy(best_harmony)
                for i in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_harmony[i] = harmony_memory[np.random.randint(self.hm_size)][i]
                        if np.random.rand() < self.par:
                            new_harmony[i] += self.bw * (np.random.rand() - 0.5) * 2
                            new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
                    else:
                        new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                
                new_harmony = self.mutate_harmony(new_harmony)
                new_value = func(new_harmony)
                evaluations += 1

                if new_value < np.max(harmony_values):
                    worst_index = np.argmax(harmony_values)
                    harmony_memory[worst_index] = new_harmony
                    harmony_values[worst_index] = new_value

                if new_value < func(best_harmony):
                    best_harmony = new_harmony

                if evaluations >= self.budget:
                    break

        return best_harmony