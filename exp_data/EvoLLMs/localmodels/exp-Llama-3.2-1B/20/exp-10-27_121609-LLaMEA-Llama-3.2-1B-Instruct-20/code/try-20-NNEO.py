import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        def mutate(individual):
            return (individual + random.uniform(-5.0, 5.0)) / 2

        def mutate_bounds(individual):
            return (individual.min() - 5.0, individual.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(self.population[i])
                if random.random() < 0.2:
                    individual = mutate(individual)
                if random.random() < 0.2:
                    individual = mutate_bounds(individual)

            new_individual = evaluate_fitness(self.population[i])
            if new_individual is not None:
                self.population[i] = new_individual

        return self.fitnesses

# NNEO Algorithm with Refining Strategy
class NNEORefining:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        def mutate(individual):
            return (individual + random.uniform(-5.0, 5.0)) / 2

        def mutate_bounds(individual):
            return (individual.min() - 5.0, individual.max() + 5.0)

        def refine_individual(individual):
            if random.random() < 0.5:
                individual = mutate(individual)
            if random.random() < 0.5:
                individual = mutate_bounds(individual)

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(self.population[i])
                if random.random() < 0.2:
                    individual = mutate(individual)
                if random.random() < 0.2:
                    individual = mutate_bounds(individual)
                individual = refine_individual(individual)

            new_individual = evaluate_fitness(self.population[i])
            if new_individual is not None:
                self.population[i] = new_individual

        return self.fitnesses

# Test the algorithms
def test_nneo(func):
    nneo = NNEO(100, 10)
    nneo_refining = NNEORefining(100, 10)
    nneo_func = func
    nneo_refining_func = func

    nneo_results = nneo(nneo_func)
    nneo_refining_results = nneo_refining(nneo_refining_func)

    print("NNEO Results:")
    print(nneo_results)
    print("NNEO Refining Results:")
    print(nneo_refining_results)

    # Compare the results
    if nneo_results[0] < nneo_refining_results[0]:
        print("NNEO is better")
    elif nneo_results[0] > nneo_refining_results[0]:
        print("NNEO Refining is better")
    else:
        print("Both algorithms are equally good")

# Test the algorithms with BBOB test suite
test_bbb = test_nneo
test_bbb_bbb = test_nneoRefining
test_bbb_bbb_bbb = test_nneoRefiningRefining
test_bbb_bbb_bbb_bbb = test_nneoRefiningRefiningRefining
test_bbb_bbb_bbb_bbb_bbb = test_nneoRefiningRefiningRefiningRefining

# Print the results
print("# Description: NNEO Algorithm with Refining Strategy")
print("# Code: ")
print("# ```python")
print("# NNEO Algorithm with Refining Strategy")
print("# Code: ")
print("# ```python")
print("# Description: NNEO Refining Algorithm")
print("# Code: ")
print("# ```python")
print("# Description: BBOB Test Suite")
print("# Code: ")
print("# ```python")
print("# Test the algorithms")
print("# Code: ")
print("# ```python")
print("# Test the algorithms with BBOB test suite")
print("# Code: ")
print("# ```python")