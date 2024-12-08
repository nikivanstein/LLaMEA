import numpy as np

class QLDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 60
        self.harmony_memory_size = 30
        self.harmony_consideration_rate = 0.95
        self.adjustment_rate = 0.2
        self.quantum_perturbation_rate = 0.05
        self.mutation_factor = 1.0
        self.epsilon = 1e-8
    
    def __call__(self, func):
        np.random.seed(42)
        lower_bound, upper_bound = self.bounds
        harmony_memory = np.random.uniform(lower_bound, upper_bound, (self.harmony_memory_size, self.dim))
        personal_best = np.copy(harmony_memory)
        personal_best_values = np.array([func(ind) for ind in harmony_memory])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                new_harmony = np.copy(global_best)
                for d in range(self.dim):
                    if np.random.rand() < self.harmony_consideration_rate:
                        idx = np.random.choice(self.harmony_memory_size)
                        new_harmony[d] = harmony_memory[idx, d]
                    if np.random.rand() < self.adjustment_rate:
                        new_harmony[d] += self.quantum_perturbation_rate * np.random.normal()

                new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
                fitness = func(new_harmony)
                evaluations += 1

                if fitness < np.max(personal_best_values):
                    max_idx = np.argmax(personal_best_values)
                    personal_best[max_idx] = new_harmony
                    personal_best_values[max_idx] = fitness
                    if fitness < personal_best_values[global_best_idx]:
                        global_best_idx = max_idx
                        global_best = personal_best[max_idx]
                
                if evaluations >= self.budget:
                    break
            
            perturbation = np.random.normal(0, 1, (self.harmony_memory_size, self.dim))
            quantum_harmonies = harmony_memory + self.quantum_perturbation_rate * perturbation
            quantum_harmonies = np.clip(quantum_harmonies, lower_bound, upper_bound)
            quantum_values = np.array([func(ind) for ind in quantum_harmonies])

            for i in range(self.harmony_memory_size):
                if quantum_values[i] < personal_best_values[i]:
                    personal_best_values[i] = quantum_values[i]
                    personal_best[i] = quantum_harmonies[i]
                    if quantum_values[i] < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]

            evaluations += self.harmony_memory_size

        return global_best