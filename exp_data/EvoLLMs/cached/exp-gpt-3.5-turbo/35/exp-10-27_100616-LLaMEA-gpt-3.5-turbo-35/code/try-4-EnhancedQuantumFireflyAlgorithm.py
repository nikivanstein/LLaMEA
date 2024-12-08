import numpy as np

class EnhancedQuantumFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.5  # Quantum probability factor

    def __call__(self, func):
        def attractiveness(i, j):
            return np.exp(-self.alpha * np.linalg.norm(population[i] - population[j]))

        def move_with_fireflies(i):
            for j in range(len(population)):
                if func(population[i]) > func(population[j]) and attractiveness(i, j) > np.random.rand():
                    population[i] += np.random.uniform(-1, 1, self.dim) * np.exp(-self.alpha * np.linalg.norm(population[j] - population[i]))

        def differential_evolution(i):
            r1, r2, r3 = np.random.choice(range(len(population)), 3, replace=False)
            mutant = population[r1] + 0.5 * (population[r2] - population[r3])
            trial = population[i] + np.random.uniform(0, 1, self.dim) * (mutant - population[i])
            if func(trial) < func(population[i]):
                population[i] = trial

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(len(population)):
                move_with_fireflies(i)
                if np.random.rand() < 0.35:
                    differential_evolution(i)

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution