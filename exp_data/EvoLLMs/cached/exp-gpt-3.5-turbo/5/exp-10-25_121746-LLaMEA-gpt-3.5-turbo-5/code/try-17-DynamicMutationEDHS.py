import numpy as np

class DynamicMutationEDHS(EDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.hmcr = np.random.uniform(0.7, 0.9)
        self.alpha = np.random.uniform(0.7, 0.9)
        self.mutation_rate = np.random.uniform(0.2, 0.4)

    def __call__(self, func):
        fitness_history = []
        for _ in range(self.budget):
            fitness = func(self.best_solution)
            fitness_history.append(fitness)
            self.update_harmony_memory()
            self.update_best_solution(fitness)
            self.update_mutation_rate(fitness_history)
        return self.best_solution