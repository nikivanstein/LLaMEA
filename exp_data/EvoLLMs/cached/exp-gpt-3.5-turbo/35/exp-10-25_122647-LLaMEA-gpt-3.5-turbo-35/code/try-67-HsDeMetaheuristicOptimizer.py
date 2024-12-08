import numpy as np

class HsDeMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        # Initialization
        harmony_memory_size = 10
        harmony_memory = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(harmony_memory_size)]
        harmony_memory_fitness = [func(harmony) for harmony in harmony_memory]
        
        for _ in range(self.budget // harmony_memory_size):
            # Harmony Search
            new_harmony = np.mean(harmony_memory, axis=0)
            
            # Differential Evolution
            trial_vector = best_solution + 0.5 * (harmony_memory[np.random.randint(harmony_memory_size)] - harmony_memory[np.random.randint(harmony_memory_size)])
            trial_fitness = func(trial_vector)
            
            if trial_fitness < best_fitness:
                best_solution = trial_vector
                best_fitness = trial_fitness
            
            # Update Harmony Memory
            min_fitness_idx = np.argmin(harmony_memory_fitness)
            if trial_fitness < harmony_memory_fitness[min_fitness_idx]:
                harmony_memory[min_fitness_idx] = trial_vector
                harmony_memory_fitness[min_fitness_idx] = trial_fitness
        
        return best_solution