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

    def _select(self, population, budget):
        # Select the best individual based on the fitness
        selected_individuals = np.random.choice(population, budget, replace=False)
        selected_individuals = [individual for individual in selected_individuals if self.f(individual, self.logger) >= self.f(self.best_individual, self.logger)]
        return selected_individuals

    def _mutate(self, selected_individuals, mutation_rate):
        # Randomly mutate the selected individuals
        mutated_individuals = []
        for individual in selected_individuals:
            mutated_individual = individual + np.random.uniform(-1, 1, self.dim)
            mutated_individuals.append(mutated_individual)
        return mutated_individuals

    def _crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.concatenate((parent1[:self.budget//2], parent2[self.budget//2:]))
        return child

    def _evaluate(self, individual):
        # Evaluate the fitness of an individual
        func_value = self.f(individual, self.logger)
        return func_value

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = self._evaluate(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Initialize the population
        self.population = [x for x in self.search_space]

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Select the best individual based on the fitness
            selected_individuals = self._select(self.population, self.budget)
            selected_individuals = self._select(selected_individuals, self.budget)

            # Perform crossover and mutation
            mutated_individuals = self._mutate(selected_individuals, 0.1)
            child_individuals = self._crossover(mutated_individuals, mutated_individuals)

            # Evaluate the fitness of the child individuals
            fitness_values = [self._evaluate(individual) for individual in child_individuals]
            child_fitness = np.mean(fitness_values)

            # Select the best child individual based on the fitness
            selected_child_individuals = self._select(child_individuals, self.budget)
            selected_child_individuals = self._select(selected_child_individuals, self.budget)

            # Perform crossover and mutation
            mutated_child_individuals = self._mutate(selected_child_individuals, 0.1)
            child_individuals = self._crossover(mutated_child_individuals, mutated_child_individuals)

            # Evaluate the fitness of the child individuals
            fitness_values = [self._evaluate(individual) for individual in child_individuals]
            child_fitness = np.mean(fitness_values)

            # Select the best child individual based on the fitness
            selected_child_individuals = self._select(child_individuals, self.budget)

            # Update the best solution and best function value
            if child_fitness > best_func_value:
                best_x = child_individuals[np.argmin(fitness_values)]
                best_func_value = child_fitness

            # Update the population
            self.population = child_individuals
            print(f"Best solution: {best_x}, Best function value: {best_func_value}")

            # Evaluate the best solution at the final search point
            func_value = self._evaluate(best_x)
            print(f"Best solution: {best_x}, Best function value: {func_value}")

        # Evaluate the best solution at the final search point
        func_value = self._evaluate(best_x)
        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
bboo.optimize(func)

# Plot the best solution and best function value
plt.plot(bboo.population, bboo.best_individual, label='Best Solution')
plt.plot(bboo.population, bboo.best_func_value, label='Best Function Value')
plt.xlabel('Individual Index')
plt.ylabel('Fitness Value')
plt.title('AdaptiveBBOO Optimization')
plt.legend()
plt.show()