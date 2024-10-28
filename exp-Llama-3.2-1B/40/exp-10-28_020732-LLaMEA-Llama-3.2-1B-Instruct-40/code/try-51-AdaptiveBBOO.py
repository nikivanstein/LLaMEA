import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.population = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def _select(self, fitnesses):
        idx = np.random.choice(len(fitnesses), self.budget, replace=False)
        return [fitnesses[i] for i in idx]

    def _crossover(self, parent1, parent2):
        x1, x2 = parent1
        x3, x4 = parent2
        crossover_idx = np.random.randint(0, len(x1), self.budget)
        for i in crossover_idx:
            x3[i], x4[i] = x1[i], x2[i]
        return x1, x3, x2, x4

    def _mutate(self, individual, mutation_rate):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < mutation_rate:
                mutated_individual[i] += np.random.uniform(-1, 1, self.dim)
        return mutated_individual

    def _evaluate(self, individual):
        func_value = individual
        for func in self.func_evaluations:
            func_value = func(individual)
        return func_value

    def _select_and_crossover(self, population):
        selected_population = self._select(self._evaluate(population))
        new_population = []
        for _ in range(self.budget):
            parent1, parent2 = selected_population[np.random.choice(len(selected_population), self.budget, replace=False)]
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child1, 0.2))
            new_population.append(self._mutate(child2, 0.2))
        return new_population

    def optimize(self, func):
        population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]
        population = self._select_and_crossover(population)
        best_individual = population[0]
        best_func_value = self._evaluate(best_individual)

        print(f"Best solution: {best_individual}, Best function value: {best_func_value}")

        return best_individual, best_func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
bboo.optimize(func)

# Plot the best solution and its function value
x = np.linspace(-5.0, 5.0, 100)
y = np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)
plt.plot(x, y, label='Best solution')
plt.plot(x, func(x), label='Function value')
plt.legend()
plt.show()