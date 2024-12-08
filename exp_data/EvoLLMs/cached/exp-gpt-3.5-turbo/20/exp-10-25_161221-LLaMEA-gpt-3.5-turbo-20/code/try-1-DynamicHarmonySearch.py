import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __call__(self, func):
        def dynamic_improvise_harmony(pop_size, hm, bw):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[i] = hm[np.random.choice(pop_size)][i]
                    if np.random.rand() < self.bw:
                        new_solution[i] = new_solution[i] + np.random.normal(0, 1)
                else:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
            return new_solution

        pop_size = 10
        harmony_memory = self.initialize_harmony_memory(pop_size)
        fitness = np.apply_along_axis(func, 1, harmony_memory)
        
        for _ in range(self.budget - pop_size):
            new_solution = dynamic_improvise_harmony(pop_size, harmony_memory, self.bw)
            new_fitness = func(new_solution)
            if new_fitness < np.max(fitness):
                idx = np.argmax(fitness)
                harmony_memory[idx] = new_solution
                fitness[idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        best_solution = harmony_memory[best_idx]

        return best_solution