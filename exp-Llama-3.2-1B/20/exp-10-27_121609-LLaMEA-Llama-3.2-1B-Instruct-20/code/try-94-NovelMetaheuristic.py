# Description: Novel metaheuristic algorithm for solving black box optimization problems
# Code: 
# ```python
import numpy as np

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.selection_prob = 0.2

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutation(x):
            new_x = np.random.uniform(x.min() - 1.0, x.max() + 1.0)
            return new_x

        def crossover(parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.selection_prob:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child

        def selection(population):
            return np.random.choice(len(population), self.population_size, replace=False)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

            # Select the best individual
            selected_individuals = selection(population)
            selected_individuals = np.array(selected_individuals)

            # Perform crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1 = selected_individuals[np.random.choice(len(selected_individuals))]
                parent2 = selected_individuals[np.random.choice(len(selected_individuals))]
                child = crossover(parent1, parent2)
                new_population.append(mutation(child))

            # Replace the old population with the new one
            self.population = new_population
            self.population_history.append(self.population)

            # Evaluate the new population
            new_fitnesses = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                new_fitnesses[i, x] = fitness

            # Update the best individual
            best_individual_index = np.argmax(new_fitnesses)
            self.population[best_individual_index] = self.population_history[-1][best_individual_index]

        return self.fitnesses

# One-line description with the main idea
# NovelMetaheuristic: A novel metaheuristic algorithm that combines crossover, mutation, and selection to optimize black box functions.
# Code: 
# ```python
# NovelMetaheuristic: A novel metaheuristic algorithm that combines crossover, mutation, and selection to optimize black box functions.
# ```