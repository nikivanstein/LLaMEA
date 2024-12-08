import random
import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None

    def __call__(self, func):
        if self.func is None:
            raise ValueError("func() must be called before calling __call__")

        # Initialize search space
        self.search_space = [(-5.0, 5.0)] * self.dim

        # Randomly select an initial point
        self.x0 = random.uniform(self.search_space)

        # Initialize population
        self.population = [self.x0]

        # Evaluate function at each point in population
        for _ in range(self.budget):
            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]

            # Select parents using tournament selection
            parents = []
            for _ in range(self.dim):
                for i in range(self.population):
                    if i == 0:
                        parents.append(self.population[i])
                    else:
                        idx = random.randint(0, len(self.population) - 1)
                        if func_values[i] > func_values[idx]:
                            parents.append(self.population[idx])

            # Crossover (recombination)
            self.population = []
            for _ in range(self.dim):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                self.population.append(parents[(parent1 + parent2) // 2])

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < 0.1:
                    self.population[i] += random.uniform(-1.0, 1.0)

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        return best_individual, best_func_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization on BBOB test suite
# using tournament selection, recombination, and mutation to search for the optimal solution

class BBOBMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.population = None
        self.best_individual = None
        self.best_func_value = float('-inf')

    def __call__(self, func):
        if self.func is None:
            raise ValueError("func() must be called before calling __call__")

        # Initialize search space
        self.search_space = [(-5.0, 5.0)] * self.dim

        # Randomly select an initial point
        self.x0 = random.uniform(self.search_space)

        # Initialize population
        self.population = [self.x0]

        # Evaluate function at each point in population
        for _ in range(self.budget):
            # Evaluate function at each point in population
            func_values = [func(self.x) for self.x in self.population]

            # Select parents using tournament selection
            parents = []
            for _ in range(self.dim):
                for i in range(self.population):
                    if i == 0:
                        parents.append(self.population[i])
                    else:
                        idx = random.randint(0, len(self.population) - 1)
                        if func_values[i] > func_values[idx]:
                            parents.append(self.population[idx])

            # Crossover (recombination)
            self.population = []
            for _ in range(self.dim):
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                self.population.append(parents[(parent1 + parent2) // 2])

            # Mutate (perturbation)
            for i in range(self.dim):
                if random.random() < 0.1:
                    self.population[i] += random.uniform(-1.0, 1.0)

        # Return best individual
        best_individual = self.population[0]
        best_func_value = func(best_individual)
        for individual in self.population:
            func_value = func(individual)
            if func_value < best_func_value:
                best_individual = individual
                best_func_value = func_value

        # Update best individual and best function value
        self.best_individual = best_individual
        self.best_func_value = best_func_value

        # Return best individual and best function value
        return best_individual, best_func_value

# Exception handling code
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return -inf
    return wrapper

# BBOB test suite evaluation code
def evaluate_bbob(func, budget, dim):
    algorithm = BBOBMetaheuristic(budget, dim)
    best_individual, best_func_value = algorithm(func)
    return best_individual, best_func_value

# Usage
if __name__ == "__main__":
    # Evaluate the BBOB test suite
    func = lambda x: x**2
    budget = 100
    dim = 10
    best_individual, best_func_value = evaluate_bbob(func, budget, dim)

    # Update the algorithm
    @exception_handler
    def update_algorithm(func):
        return BBOBMetaheuristicAlgorithm(budget, dim)

    # Update the algorithm
    updated_algorithm = update_algorithm(func)

    # Optimize the function using the updated algorithm
    best_individual, best_func_value = updated_algorithm(func)
    print(f"Best individual: {best_individual}")
    print(f"Best function value: {best_func_value}")