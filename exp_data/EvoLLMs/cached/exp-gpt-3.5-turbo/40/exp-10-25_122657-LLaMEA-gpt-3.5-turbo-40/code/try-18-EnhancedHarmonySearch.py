import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(harmony_memory):
            return np.array([func(solution) for solution in harmony_memory])

        def update_harmony_memory(harmony_memory, fitness):
            worst_idx = np.argmax(fitness)
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            harmony_memory[worst_idx] = new_solution
            return harmony_memory

        def differential_evolution(harmony_memory, fitness):
            best_idx = np.argmin(fitness)
            best_solution = harmony_memory[best_idx]
            F = 0.5  # Differential weight
            CR = 0.5  # Crossover rate

            for i in range(self.budget // 10):
                base_idx = np.random.choice(self.budget)
                target_idx = np.random.choice(self.budget)
                while target_idx == base_idx:
                    target_idx = np.random.choice(self.budget)

                base = harmony_memory[base_idx]
                target = harmony_memory[target_idx]
                mutant = base + F * (best_solution - target)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, harmony_memory[i])
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    harmony_memory[i] = trial
                    fitness[i] = trial_fitness

            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)

        harmony_memory = differential_evolution(harmony_memory, fitness)
        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]