import numpy as np
import concurrent.futures

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def evaluate_solution(self, func, harmony):
        return func(harmony)

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            harmony_memory_fitness = np.array(list(executor.map(lambda harmony: self.evaluate_solution(func, harmony), harmony_memory)))

            for _ in range(int(self.budget * 0.8)):  # Adjusted parallel evaluation percentage
                new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                new_harmony_fitness = func(new_harmony)
                
                if new_harmony_fitness < harmony_memory_fitness.max():
                    index = harmony_memory_fitness.argmax()
                    harmony_memory[index] = new_harmony
                    harmony_memory_fitness[index] = new_harmony_fitness
        
        best_harmony = harmony_memory[np.argmin(harmony_memory_fitness)]
        return best_harmony