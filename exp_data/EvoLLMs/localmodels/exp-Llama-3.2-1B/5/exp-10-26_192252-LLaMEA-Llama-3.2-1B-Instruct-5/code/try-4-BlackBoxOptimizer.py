import random
import numpy as np
import operator

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5
        self.population_history = []

    def __call__(self, func):
        def func_eval(x):
            return func(x)

        # Initialize the population with random points in the search space
        x = np.random.uniform(-5.0, 5.0, self.dim)
        population = [x] * self.population_size

        # Evaluate the function for each point in the population
        for _ in range(self.budget):
            # Select the fittest points to reproduce
            fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

            # Create new offspring by crossover and mutation
            offspring = []
            for i in range(self.population_size // 2):
                parent1, parent2 = random.sample(fittest_points, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child += random.uniform(-5.0, 5.0)
                offspring.append(child)

            # Replace the worst points in the population with the new offspring
            population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Select the fittest points to reproduce
        fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

        # Create new offspring by crossover and mutation
        offspring = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.sample(fittest_points, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the worst points in the population with the new offspring
        population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Select the fittest points to reproduce
        fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

        # Create new offspring by crossover and mutation
        offspring = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.sample(fittest_points, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the worst points in the population with the new offspring
        population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Update the population history
        self.population_history.append((population, func_eval(population[0])))

        return population

    def update(self, func, budget):
        best_individual, best_fitness = self.population_history[-1]
        new_population = self.__call__(func)
        new_fitness = func(new_population[0])

        # Update the population history
        self.population_history.append((new_population, new_fitness))

        # Update the best individual
        if new_fitness > best_fitness:
            best_individual, best_fitness = new_population, new_fitness

        # Refine the strategy using probability 0.05
        if random.random() < 0.05:
            new_individual = best_individual
            for _ in range(self.budget):
                # Select the fittest points to reproduce
                fittest_points = sorted(best_individual, key=func_eval, reverse=True)[:self.population_size // 2]

                # Create new offspring by crossover and mutation
                offspring = []
                for i in range(self.population_size // 2):
                    parent1, parent2 = random.sample(fittest_points, 2)
                    child = (parent1 + parent2) / 2
                    if random.random() < self.mutation_rate:
                        child += random.uniform(-5.0, 5.0)
                    offspring.append(child)

                # Replace the worst points in the population with the new offspring
                best_individual = [x if func_eval(x) < func_eval(p) else p for p in best_individual]

                # Select the fittest points to reproduce
                best_points = sorted(best_individual, key=func_eval, reverse=True)[:self.population_size // 2]

                # Create new offspring by crossover and mutation
                offspring = []
                for i in range(self.population_size // 2):
                    parent1, parent2 = random.sample(best_points, 2)
                    child = (parent1 + parent2) / 2
                    if random.random() < self.mutation_rate:
                        child += random.uniform(-5.0, 5.0)
                    offspring.append(child)

                # Replace the worst points in the population with the new offspring
                best_individual = [x if func_eval(x) < func_eval(p) else p for p in best_individual]

            # Update the best individual
            best_individual, best_fitness = new_population, new_fitness

        return best_individual, best_fitness