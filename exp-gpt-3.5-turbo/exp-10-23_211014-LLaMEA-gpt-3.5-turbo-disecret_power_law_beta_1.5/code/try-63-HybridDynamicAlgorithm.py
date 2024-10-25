import numpy as np

class HybridDynamicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.alpha = 0.5
        self.gamma = 0.1
        self.inertia_weight = 0.5
        self.c1 = 2.0
        self.c2 = 2.0

    def _differential_evolution(self, population, func):
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.gamma * (b - c), -5.0, 5.0)
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.alpha, mutant, population[i])
            if func(trial) < func(population[i]):
                population[i] = trial
        return population

    def _firefly_algorithm(self, population, func):
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        attractiveness = 1 / (1 + np.sqrt(np.sum((population - best_solution) ** 2, axis=1)))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if fitness[i] > fitness[j]:
                    population[i] += attractiveness[j] * (population[j] - population[i]) + np.random.uniform(-0.1, 0.1, self.dim)
                    population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def _pso(self, population, func):
        velocities = np.zeros((self.population_size, self.dim))
        best_individual = population[np.argmin([func(individual) for individual in population])]
        global_best = best_individual.copy()
        for i in range(self.population_size):
            velocities[i] = self.inertia_weight * velocities[i] + self.c1 * np.random.rand() * (best_individual - population[i]) + self.c2 * np.random.rand() * (global_best - population[i])
            population[i] += velocities[i]
            population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def _levy_flight(self, population):
        levy = np.random.standard_cauchy(size=(self.population_size, self.dim))
        population += 0.01 * levy
        population = np.clip(population, -5.0, 5.0)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._differential_evolution(population, func)
            population = self._firefly_algorithm(population, func)
            population = self._pso(population, func)
            population = self._levy_flight(population)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution