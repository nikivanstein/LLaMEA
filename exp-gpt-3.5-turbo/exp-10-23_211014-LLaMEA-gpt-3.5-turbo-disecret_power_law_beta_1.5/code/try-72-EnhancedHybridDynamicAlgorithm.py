import numpy as np

class EnhancedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.probability = 0.016129032258064516

    def _fine_tune_de_component(self, population, func):
        for i in range(self.population_size):
            if np.random.rand() < self.probability:
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.gamma * (b - c), -5.0, 5.0)
                trial = np.where(np.random.uniform(0, 1, self.dim) < self.alpha, mutant, population[i])
                if func(trial) < func(population[i]):
                    population[i] = trial
        return population

    def _fine_tune_fa_component(self, population, func):
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        attractiveness = 1 / (1 + np.sqrt(np.sum((population - best_solution) ** 2, axis=1)))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if fitness[i] > fitness[j] and np.random.rand() < self.probability:
                    population[i] += attractiveness[j] * (population[j] - population[i]) + np.random.uniform(-0.1, 0.1, self.dim)
                    population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._fine_tune_de_component(population, func)
            population = self._fine_tune_fa_component(population, func)
            population = self._pso(population, func)
            population = self._levy_flight(population)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution