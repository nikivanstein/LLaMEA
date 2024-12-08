import numpy as np
import random
from functools import wraps

class HybridEvoAlg:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(solution)
        return population

    def fitness(self, solution, func):
        return func(*solution)

    def selection(self, population, func):
        fitnesses = [self.fitness(solution, func) for solution in population]
        fitnesses = np.array(fitnesses)
        fitness_min = np.min(fitnesses)
        fitness_max = np.max(fitnesses)
        selection_probabilities = (fitnesses - fitness_min) / (fitness_max - fitness_min)
        selected_indices = np.random.choice(len(population), size=self.population_size, p=selection_probabilities)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        child = [0.5 * (parent1[i] + parent2[i]) for i in range(self.dim)]
        return child

    def mutation(self, solution):
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += random.uniform(-1.0, 1.0)
                solution[i] = max(-5.0, min(5.0, solution[i]))
        return solution

    def local_search(self, solution, func):
        best_solution = solution
        for i in range(self.dim):
            for new_solution in [solution[:i] + [solution[i] + 1.0] + solution[i+1:],
                                solution[:i] + [solution[i] - 1.0] + solution[i+1:]]:
                fitness = self.fitness(new_solution, func)
                if fitness < self.fitness(best_solution, func):
                    best_solution = new_solution
        return best_solution

    def hybrid_evo_alg(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(self.budget):
                population = self.selection(self.population, func)
                new_population = []
                for _ in range(self.population_size):
                    parent1, parent2 = random.sample(population, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutation(child)
                    new_population.append(child)
                self.population = new_population
                best_solution = max(self.population, key=self.fitness)
                if self.fitness(best_solution, func) < self.best_fitness:
                    self.best_solution = best_solution
                    self.best_fitness = self.fitness(best_solution, func)
                if self.best_fitness < func(0):
                    return self.best_solution
            return self.best_solution
        return wrapper

# Usage
if __name__ == "__main__":
    from BBOB import bbo_bench
    from numpy import random
    # Define the functions to be optimized
    functions = {
        "f1": lambda x: sum([i**2 for i in x]),
        "f2": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f3": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f4": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f5": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f6": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f7": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f8": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f9": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f10": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f11": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f12": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f13": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f14": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f15": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f16": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f17": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]),
        "f18": lambda x: sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x]) + sum([i**2 for i in x])
    }

    # Define the BBOB test suite
    test_suite = {
        "f1": "f1",
        "f2": "f2",
        "f3": "f3",
        "f4": "f4",
        "f5": "f5",
        "f6": "f6",
        "f7": "f7",
        "f8": "f8",
        "f9": "f9",
        "f10": "f10",
        "f11": "f11",
        "f12": "f12",
        "f13": "f13",
        "f14": "f14",
        "f15": "f15",
        "f16": "f16",
        "f17": "f17",
        "f18": "f18"
    }

    # Run the optimization algorithm
    for func_name, func in functions.items():
        print(f"Optimizing function {func_name}...")
        algorithm = HybridEvoAlg(budget=100, dim=10)
        solution = algorithm(func_name)
        print(f"Optimized solution: {solution}")
        print(f"Optimized fitness: {algorithm.fitness(solution, func)}")
        print()

# Evaluate the algorithm on the BBOB test suite
for func_name, func in test_suite.items():
    print(f"Evaluating function {func_name}...")
    algorithm = HybridEvoAlg(budget=100, dim=10)
    solution = algorithm(func)
    print(f"Optimized solution: {solution}")
    print(f"Optimized fitness: {algorithm.fitness(solution, func)}")
    print()
