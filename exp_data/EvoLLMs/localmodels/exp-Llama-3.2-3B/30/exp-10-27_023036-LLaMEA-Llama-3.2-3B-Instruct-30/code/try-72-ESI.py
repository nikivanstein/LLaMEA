import numpy as np
import random

class ESI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.swarm_size = 10
        self.initial_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_individual = self.initial_population[0]
        self.best_score = self.evaluate_function(self.best_individual, self.func)

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the best individual in the current population
            score = self.evaluate_function(self.best_individual, func)
            print(f"Best individual: {self.best_individual}, Best score: {score}")

            # Create a new population by combining the current population with a swarm of random individuals
            new_population = np.concatenate((self.initial_population, np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)).reshape(-1, self.dim)))

            # Apply evolutionary strategies to the new population
            for i in range(self.population_size):
                # Select a parent from the current population
                parent = np.random.choice(self.population_size, 1, p=np.array(self.evaluate_function(new_population, func)) / self.best_score)

                # Select a child from the swarm
                child = new_population[i]

                # Apply crossover and mutation to the child
                child = self.crossover(child, new_population[parent, :])
                child = self.mutation(child)

                # Replace the worst individual in the new population with the child
                new_population[i] = child

            # Update the best individual in the new population
            self.best_individual = self.get_best_individual(new_population)

            # Update the best score
            self.best_score = self.evaluate_function(self.best_individual, func)

    def evaluate_function(self, x, func):
        return func(x)

    def crossover(self, parent, child):
        # Apply a random crossover strategy
        if random.random() < 0.5:
            return (parent + child) / 2
        else:
            return parent

    def mutation(self, x):
        # Apply a random mutation strategy
        if random.random() < 0.1:
            return x + np.random.uniform(-1.0, 1.0, self.dim)
        else:
            return x

    def get_best_individual(self, population):
        return population[np.argmax([self.evaluate_function(individual, self.func) for individual in population])]

# Example usage:
def func(x):
    return np.sum(x**2)

esi = ESI(100, 2)
esi(func)