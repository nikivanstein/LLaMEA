# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: import random
# import numpy as np
# import copy

class LLaMEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(self.budget):
            individual = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
            self.population.append(copy.deepcopy(individual))

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Assign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Update the function values for the next iteration
        for _ in range(self.budget):
            new_centers = self.cluster_centers.copy()
            for j in range(self.dim):
                for k in range(self.dim):
                    new_centers[j, k] += 0.1 * (self.cluster_centers[j, k] - self.cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (self.cluster_centers[j, k] - self.cluster_centers[j, k] ** 2))
            self.cluster_centers = new_centers

        # Reassign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        self.func_values[func.__name__] = best_value

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        j, k = random.sample(range(self.dim), 2)
        individual[j], individual[k] = individual[k], individual[j]
        return individual

    def evaluate_fitness(self, individual):
        func_value = func(individual)
        return func_value

# Description: Gradient Descent with Stochastic Gradient Clustering
# Code: import random
# import numpy as np
# import copy

class GDSC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(self.budget):
            individual = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
            self.population.append(copy.deepcopy(individual))

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Assign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Update the function values for the next iteration
        for _ in range(self.budget):
            new_centers = self.cluster_centers.copy()
            for j in range(self.dim):
                for k in range(self.dim):
                    new_centers[j, k] += 0.1 * (self.cluster_centers[j, k] - self.cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (self.cluster_centers[j, k] - self.cluster_centers[j, k] ** 2))
            self.cluster_centers = new_centers

        # Reassign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        self.func_values[func.__name__] = best_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        return best_individual

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        j, k = random.sample(range(self.dim), 2)
        individual[j], individual[k] = individual[k], individual[j]
        return individual

# Description: Gradient Descent with Stochastic Gradient Clustering
# Code: import random
# import numpy as np
# import copy

class LLaMEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(self.budget):
            individual = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
            self.population.append(copy.deepcopy(individual))

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Assign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Update the function values for the next iteration
        for _ in range(self.budget):
            new_centers = self.cluster_centers.copy()
            for j in range(self.dim):
                for k in range(self.dim):
                    new_centers[j, k] += 0.1 * (self.cluster_centers[j, k] - self.cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (self.cluster_centers[j, k] - self.cluster_centers[j, k] ** 2))
            self.cluster_centers = new_centers

        # Reassign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        self.func_values[func.__name__] = best_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        return best_individual

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        j, k = random.sample(range(self.dim), 2)
        individual[j], individual[k] = individual[k], individual[j]
        return individual

# Description: Gradient Descent with Stochastic Gradient Clustering
# Code: import random
# import numpy as np
# import copy

class LLaMEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.cluster_centers = None
        self.population = []

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(self.budget):
            individual = np.random.uniform(self.search_space[0], self.search_space[1], (self.dim, self.dim))
            self.population.append(copy.deepcopy(individual))

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Assign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Update the function values for the next iteration
        for _ in range(self.budget):
            new_centers = self.cluster_centers.copy()
            for j in range(self.dim):
                for k in range(self.dim):
                    new_centers[j, k] += 0.1 * (self.cluster_centers[j, k] - self.cluster_centers[j, k] * (func(self.func_values[sample]) - self.func_values[sample][j, k]) / (self.cluster_centers[j, k] - self.cluster_centers[j, k] ** 2))
            self.cluster_centers = new_centers

        # Reassign each individual to the closest cluster center
        self.cluster_centers = np.array([self.cluster_centers])
        for sample in func.__code__.co_varnames[1:]:
            dist = np.linalg.norm(func(self.func_values[sample]) - self.cluster_centers, axis=1)
            self.cluster_centers = np.argmin(dist, axis=0)

        # Evaluate the function for each individual
        for individual in self.population:
            func_value = func(individual)
            self.func_values[func.__name__] = func_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        self.func_values[func.__name__] = best_value

        # Return the best individual
        best_individual = copy.deepcopy(self.population[0])
        best_value = func(best_individual)
        for individual in self.population:
            individual_value = func(individual)
            if individual_value > best_value:
                best_individual = individual
                best_value = individual_value
        return best_individual

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        j, k = random.sample(range(self.dim), 2)
        individual[j], individual[k] = individual[k], individual[j]
        return individual

# Description: Gradient Descent with Stochastic Gradient Clustering
# Code: import random
# import numpy as np
# import copy

def BBOB(func, budget, dim, search_space, num_samples):
    algorithm = LLaMEA(budget, dim)
    for _ in range(num_samples):
        algorithm(__call__(func))
    return algorithm.func_values

# Example usage:
def sphere(x):
    return np.sum(x ** 2)

# Generate 1000 random functions
functions = [sphere(np.random.uniform(-10, 10, dim)) for _ in range(1000)]

# Evaluate the functions using BBOB
best_function = max(functions, key=lambda func: BBOB(func, 100, 5, (-10, 10), 10000))
print("Best function:", best_function.__name__)
print("Best value:", BBOB(best_function, 100, 5, (-10, 10), 10000))