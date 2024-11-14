import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __evaluate_candidates(self, func, candidates):
        return [func(candidate) for candidate in candidates]
    
    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        harmony_memory_fitness = np.array([func(harmony) for harmony in harmony_memory])

        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                new_harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
                new_harmony_fitness = executor.map(lambda candidate: func(candidate), new_harmonies)

                for candidate, fitness in zip(new_harmonies, new_harmony_fitness):
                    if fitness < harmony_memory_fitness.max():
                        index = harmony_memory_fitness.argmax()
                        harmony_memory[index] = candidate
                        harmony_memory_fitness[index] = fitness

        best_harmony = harmony_memory[np.argmin(harmony_memory_fitness)]
        return best_harmony