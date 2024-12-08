import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)  # 10 dimensions

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Define a genetic algorithm to refine the solution
class GeneticBBOO(AdaptiveBBOO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        # Initialize the population with random solutions
        population = np.array([self.optimize(func) for _ in range(self.population_size)])

        return population

    def select_parents(self):
        # Select parents using tournament selection
        parents = np.array([np.random.choice(self.population_size, p=[0.5, 0.5]) for _ in range(self.population_size)])

        return parents

    def crossover(self, parents):
        # Perform crossover using uniform crossover
        offspring = np.zeros_like(parents)
        for i in range(self.population_size):
            for j in range(i+1, self.population_size):
                if np.random.rand() < 0.5:
                    offspring[i, :] = parents[i, :]
                    offspring[j, :] = parents[j, :]

        return offspring

    def mutate(self, offspring):
        # Randomly mutate the offspring
        for i in range(self.population_size):
            if np.random.rand() < 0.2:
                offspring[i, :] += np.random.uniform(-1, 1, self.dim)

        return offspring

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_value = func(individual)
        return func_value

    def __call__(self, func):
        # Initialize the population with random solutions
        self.population = np.array([self.optimize(func) for _ in range(self.population_size)])

        # Select parents using tournament selection
        parents = self.select_parents()

        # Perform crossover and mutation
        offspring = self.crossover(parents)
        offspring = self.mutate(offspring)

        # Evaluate the fitness of the offspring
        fitness = np.array([self.evaluate_fitness(individual) for individual in offspring])

        # Select the best individual
        best_individual = np.argmax(fitness)

        # Return the best individual
        return best_individual, fitness[best_individual]

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the GeneticBBOO class
gbboo = GeneticBBOO(1000, 10)  # 10 dimensions

# Optimize the function with 1000 evaluations
gbboo.optimize(func)

# Plot the results
plt.plot(np.linspace(-5, 5, 100), gbboo.func_evaluations)
plt.xlabel('Input')
plt.ylabel('Function Value')
plt.title('Black Box Optimization using Genetic Algorithm')
plt.show()