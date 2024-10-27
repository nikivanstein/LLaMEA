import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population = []
        self.population_size = 100
        self.population_mutations = 0.01
        self.population_crossovers = 0.5

    def __call__(self, func):
        # Initialize population
        for _ in range(self.population_size):
            new_individual = self.generate_new_individual()
            self.population.append(new_individual)

        while len(self.population) < self.population_size:
            # Select fittest individuals
            fittest_individuals = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.population_size // 2]

            # Create new population
            new_population = []
            for _ in range(self.population_size):
                # Select parents using tournament selection
                parent1 = random.choice(fittest_individuals)
                parent2 = random.choice(fittest_individuals)

                # Select best parent
                if random.random() < 0.5:
                    parent1 = parent1
                else:
                    parent2 = parent2

                # Crossover (recombination)
                child = self.crossover(parent1, parent2)

                # Mutate
                if random.random() < self.population_mutations:
                    child = self.mutate(child)

                new_population.append(child)

            # Replace worst individuals with new ones
            self.population = new_population

        # Evaluate the best individual
        best_individual = max(self.population, key=lambda x: x.fitness)

        # Update budget
        self.func_evaluations += best_individual.fitness

        return best_individual, best_individual.fitness

    def generate_new_individual(self):
        # Generate a random point in the search space
        point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        # Evaluate the function at the point
        func_value = self.func(point)
        # Return the point
        return point

    def crossover(self, parent1, parent2):
        # Select a random point in the search space
        crossover_point = random.uniform(self.search_space[0], self.search_space[1])

        # Split the parents
        child1 = parent1[:crossover_point]
        child2 = parent2[crossover_point:]

        # Merge the children
        child = child1 + child2

        return child

    def mutate(self, individual):
        # Flip a random bit
        if random.random() < self.population_crossovers:
            bit = random.randint(0, len(individual) - 1)
            individual[bit] = 1 - individual[bit]
        return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 