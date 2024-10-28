import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def fitness_func(x):
            return evaluate_func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [fitness_func(x) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        return self.elite[0]

    def update(self, func, budget):
        # Select a new individual
        new_individual = self.evaluate_fitness(func)

        # Refine the strategy using the probability 0.45
        new_individual = self.refine_strategy(new_individual, func, budget)

        # Replace the elite with the new individual
        self.elite = [new_individual]

        # Check if the budget is exhausted
        if len(self.elite) > self.elite_size:
            self.elite.pop()

        # Evaluate the new individual
        new_individual = self.evaluate_fitness(new_individual)

        # Update the best individual
        best_individual = max(self.elite, key=fitness_func)

        return best_individual

    def refine_strategy(self, individual, func, budget):
        # Define the mutation probability
        mutation_probability = 0.1

        # Define the crossover probability
        crossover_probability = 0.5

        # Define the mutation strategy
        def mutate(individual):
            if random.random() < mutation_probability:
                index = random.randint(0, self.dim - 1)
                individual[index] += random.uniform(-1.0, 1.0)

        # Define the crossover strategy
        def crossover(parent1, parent2):
            if random.random() < crossover_probability:
                child = (parent1 + parent2) / 2
                return child
            else:
                return parent1

        # Perform mutation and crossover
        children = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(self.elite, 2)
            child = crossover(parent1, parent2)
            children.append(mutate(child))

        # Replace the elite with the children
        self.elite = children

        return self.elite[0]