import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.hms = max(5, int(budget / (10 * dim)))  # Harmony memory size
        self.hmcr = 0.9  # Harmony memory considering rate
        self.par = 0.3  # Pitch adjusting rate
        self.dynamic_par = 0.01  # Dynamic component for PAR
        self.randomness_factor = 0.1  # Factor for randomness in adaptation

    def __call__(self, func):
        # Initialize harmony memory
        harmony_memory = np.random.uniform(self.lb, self.ub, (self.hms, self.dim))
        fitness = np.array([func(harmony) for harmony in harmony_memory])
        num_evaluations = self.hms
        
        best_idx = np.argmin(fitness)
        best_harmony = harmony_memory[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            if num_evaluations >= self.budget:
                break
            
            # Create new harmony
            new_harmony = np.empty(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = harmony_memory[np.random.randint(0, self.hms), j]
                    if np.random.rand() < self.par + self.dynamic_par * np.random.rand():
                        new_harmony[j] += self.randomness_factor * (2 * np.random.rand() - 1)
                else:
                    new_harmony[j] = np.random.uniform(self.lb, self.ub)

            new_harmony = np.clip(new_harmony, self.lb, self.ub)
            new_fitness = func(new_harmony)
            num_evaluations += 1
            
            # Update harmony memory
            if new_fitness < best_fitness:
                best_harmony = new_harmony
                best_fitness = new_fitness
                
            if new_fitness < np.max(fitness):
                worst_idx = np.argmax(fitness)
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness
        
        return best_harmony, best_fitness