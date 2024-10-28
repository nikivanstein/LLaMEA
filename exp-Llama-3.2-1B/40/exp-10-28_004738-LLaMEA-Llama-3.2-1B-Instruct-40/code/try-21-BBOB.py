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
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget, mutation_prob=0.4):
    # Initialize the population with random solutions
    population = [x0] * budget
    for _ in range(budget):
        # Select the individual to mutate
        parent1, parent2 = random.sample(population, 2)
        # Create a new individual by combining the parents
        child = (parent1 + parent2) / 2
        # Evaluate the fitness of the child
        fitness = func(child)
        # If the mutation probability is high, mutate the child
        if random.random() < mutation_prob:
            # Randomly select a mutation point
            idx = random.randint(0, self.dim - 1)
            # Swap the mutation point with a random point in the bounds
            child[idx], child[idx + 1] = child[idx + 1], child[idx]
            # Evaluate the fitness of the mutated child
            fitness = func(child)
            # If the mutation probability is high, mutate the child again
            if random.random() < mutation_prob:
                # Randomly select a mutation point
                idx = random.randint(0, self.dim - 1)
                # Swap the mutation point with a random point in the bounds
                child[idx], child[idx + 1] = child[idx + 1], child[idx]
                # Evaluate the fitness of the mutated child
                fitness = func(child)
        # Add the child to the population
        population.append(child)
    # Evaluate the fitness of each individual in the population
    fitnesses = [func(individual) for individual in population]
    # Select the fittest individuals to reproduce
    parents = random.sample(population, int(self.budget * 0.5))
    # Create a new population by combining the parents
    new_population = []
    for _ in range(budget):
        # Select a random parent
        parent1, parent2 = random.sample(parents, 2)
        # Create a new individual by combining the parents
        child = (parent1 + parent2) / 2
        # Evaluate the fitness of the child
        fitness = func(child)
        # Add the child to the new population
        new_population.append(child)
    return new_population

# Example usage:
budget = 100
dim = 5
bounds = [(-5, 5) for _ in range(dim)]
problem = BBOB(budget, dim)
best_solution = None
best_fitness = -np.inf
for _ in range(100):
    solution = bbo_opt(problem.func, [0] * dim, bounds, budget, mutation_prob=0.4)
    fitness = problem.func(solution)
    if fitness > best_fitness:
        best_fitness = fitness
        best_solution = solution
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")