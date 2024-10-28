import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class MutationExpMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def mutate(self, individual):
        # Refine the strategy by changing the individual's lines
        # with a probability of 0.45
        if random.random() < 0.45:
            line_index = random.randint(0, self.dim - 1)
            individual[line_index] = random.uniform(-5.0, 5.0)

        # Ensure the individual's lines stay within the search space
        individual = [x for x in individual if x >= -5.0 and x <= 5.0]

        return individual

class GeneticAlgorithmMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def evolve(self, population, mutation_exp):
        # Select the fittest individuals
        fittest = sorted(population, key=lambda x: x[1], reverse=True)[:self.budget]

        # Create a new population by mutating the fittest individuals
        new_population = []
        for _ in range(self.budget):
            individual = fittest.pop()
            mutated_individual = mutation_exp.mutate(individual)
            new_population.append(mutated_individual)

        # Replace the old population with the new one
        population[:] = new_population

        return new_population

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.metaheuristic = MutationExpMetaheuristic(budget, dim)
        self.population = [self.metaheuristic() for _ in range(100)]

    def __call__(self, func):
        return self.metaheuristic(func)

# Initialize the optimizer
optimizer = BlackBoxOptimizer(100, 10)

# Optimize the function
best_func = optimizer(func)
print(best_func)