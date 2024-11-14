import numpy as np

class DynamicQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness)
            elite_index = sorted_indices[0]
            elite = self.population[elite_index]
            for i in range(self.budget):
                if i != elite_index:
                    mutation = self.population[i] + np.random.uniform(-0.1, 0.1, self.dim) * self.mutation_rate
                    offspring = 0.5 * self.population[i] + 0.5 * elite
                    quantum_mutation = offspring + np.random.choice([-1, 1]) * np.random.rand() * (elite - self.population[i]) * self.mutation_rate
                    self.population[i] = quantum_mutation if func(quantum_mutation) < fitness[i] else mutation
            self.mutation_rate *= 0.99  # Update mutation rate dynamically
        best_index = np.argmin([func(individual) for individual in self.population])
        return self.population[best_index]