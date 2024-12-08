import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        return func(x0)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    # Novel heuristic algorithm: "Adaptive BBOB"
    # Description: Adaptive Black Box Optimization using BBOB
    # Code: 

    def evaluate_fitness(individual, func, bounds, budget):
        return func(individual)

    def mutate(individual, func, bounds, budget):
        # Refine the individual's strategy based on the budget
        # for the current iteration
        # and update the individual's fitness
        # with the new strategy
        # This is where the adaptive aspect comes in
        # We'll use a simple heuristic to determine the
        # new strategy: if the budget is less than 0.4, we
        # use the current strategy; otherwise, we use a
        # random strategy
        strategy = np.random.uniform(0, 1)
        if strategy < 0.4:
            # Use the current strategy
            individual = individual
        else:
            # Use a random strategy
            individual = np.random.uniform(bounds[0], bounds[1])

        # Update the individual's fitness with the new strategy
        fitness = evaluate_fitness(individual, func, bounds, budget)
        individual[0] = fitness
        return individual

    def run_algorithm():
        x0 = np.array([0.0] + [random.uniform(-5.0, 5.0) for _ in range(self.dim - 1)])
        best_individual = x0
        best_fitness = evaluate_fitness(best_individual, func, bounds, budget)
        for _ in range(self.budget):
            new_individual = mutate(best_individual, func, bounds, budget)
            new_fitness = evaluate_fitness(new_individual, func, bounds, budget)
            if new_fitness < best_fitness:
                best_individual = new_individual
                best_fitness = new_fitness
        return best_individual, best_fitness

    return bbo_opt, run_algorithm

# Initialize the BBOB algorithm
bbo_opt, run_algorithm = BBOB(100, 5)

# Run the algorithm and print the results
best_individual, best_fitness = run_algorithm()
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)