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

def bbo_opt(func, x0, bounds, budget):
    population = [x0] * budget
    for _ in range(budget):
        # Select parents using tournament selection
        parents = []
        for _ in range(2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if random.random() < 0.5:
                parent1 = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                parent2 = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                parent1 = random.uniform(bounds[0], bounds[1])
            parents.append(parent1)
        
        # Crossover
        offspring1 = parent1[:len(parent1)//2] + [parent1[len(parent1)//2] if len(parent1) % 2 == 1 else parent1[len(parent1)//2] + parent1[len(parent1)//2] for parent1 in parents]
        offspring2 = parent2[:len(parent2)//2] + [parent2[len(parent2)//2] if len(parent2) % 2 == 1 else parent2[len(parent2)//2] + parent2[len(parent2)//2] for parent2 in parents]
        
        # Mutate
        offspring1 = offspring1 + [random.uniform(bounds[0], bounds[1]) for _ in range(len(offspring1)//2)]
        offspring2 = offspring2 + [random.uniform(bounds[0], bounds[1]) for _ in range(len(offspring2)//2)]
        
        # Replace with the fittest individual
        population[0], population[budget - 1] = max(population), max(offspring1 + offspring2)
        
        # Replace with the fittest individual
        population = population[:budget]
    
    # Return the fittest individual
    return population[0]

# Novel Metaheuristic: Evolutionary Algorithm using Genetic Programming
# Code: 