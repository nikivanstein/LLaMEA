import numpy as np

class ImprovedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.6  # Increased pitch adjustment rate for faster convergence
        self.bandwidth = 0.01

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        
        def pitch_adjustment(new_solution, index):
            if np.random.rand() < self.par:
                new_solution[index] = new_solution[index] + np.random.uniform(-self.bandwidth, self.bandwidth)
                new_solution[index] = np.clip(new_solution[index], self.lower_bound, self.upper_bound)
            return new_solution
        
        harmony_memory = initialize_harmony_memory()
        fitness = np.array([func(solution) for solution in harmony_memory])
        for _ in range(self.budget - self.budget):
            new_solution = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[i] = harmony_memory[np.random.choice(self.budget)][i]
                else:
                    new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                new_solution = pitch_adjustment(new_solution, i)
            
            new_fitness = func(new_solution)
            worst_index = np.argmax(fitness)
            if new_fitness < fitness[worst_index]:
                harmony_memory[worst_index] = new_solution
                fitness[worst_index] = new_fitness
        
        best_index = np.argmin(fitness)
        return harmony_memory[best_index], fitness[best_index]