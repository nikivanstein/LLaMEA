import numpy as np

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

        def mutate(individual):
            new_individual = individual.copy()
            if np.random.rand() < 0.2:
                new_individual[np.random.randint(0, self.dim)] += np.random.uniform(-5.0, 5.0)
            return new_individual

        def crossover(parent1, parent2):
            if np.random.rand() < 0.5:
                child = parent1.copy()
                child[np.random.randint(0, self.dim)] = parent2[np.random.randint(0, self.dim)]
                return child
            else:
                child = parent1.copy()
                child[np.random.randint(0, self.dim)] = parent2[np.random.randint(0, self.dim)]
                return child

        def selection(population):
            return np.random.choice(len(population), self.population_size, replace=False)

        def __next_generation(individual, population):
            next_generation = population.copy()
            for _ in range(self.budget):
                for i in range(self.population_size):
                    x = individual[i]
                    fitness = objective(x)
                    if fitness < individual[i] + 1e-6:
                        individual[i] = x
                        next_generation[i] = x

            # Select the best individual
            selected_individual = selection(next_generation)
            selected_individual = selected_individual[np.argsort(individual)]

            # Select the worst individual
            worst_individual = selection(next_generation)
            worst_individual = worst_individual[np.argsort(individual)]

            # Refine the selected individual
            new_individual = mutate(selected_individual)
            new_individual = crossover(new_individual, worst_individual)

            # Replace the worst individual with the new individual
            next_generation[np.argsort(individual)] = new_individual

            return next_generation

        return __next_generation(self.population, population)

# One-line description with main idea
# Novel Hybrid Metaheuristic Algorithm for Black Box Optimization
# 