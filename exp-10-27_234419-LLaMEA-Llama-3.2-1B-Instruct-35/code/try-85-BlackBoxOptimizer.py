import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

def adaptive_random_search(bboo, budget, dim, population_size=100):
    """
    Adaptive Random Search with Adaptive Sampling and Local Search for Black Box Optimization
    """
    # Initialize the population
    population = [bboo.__call__(func) for func in bboo.funcs]

    # Evolve the population using adaptive random search
    for _ in range(budget):
        # Evaluate the fitness of each individual
        fitnesses = [bboo.func(individual) for individual in population]

        # Select the best individuals
        selected_indices = np.argsort(fitnesses)[-population_size:]

        # Select a new population using adaptive sampling and local search
        new_population = []
        for _ in range(population_size):
            # Select a random individual from the search space
            individual = np.random.choice(self.search_space, size=dim, replace=False)

            # If the individual is already in the population, use local search
            if individual in selected_indices:
                # Evaluate the fitness of the individual
                fitness = bboo.func(individual)

                # Perform local search
                best_individual = None
                best_fitness = np.inf
                for _ in range(10):
                    # Generate a new individual by perturbing the current individual
                    perturbed_individual = individual + np.random.normal(0, 1, dim)

                    # Evaluate the fitness of the perturbed individual
                    fitness = bboo.func(perturbed_individual)

                    # If the fitness is better than the current best fitness, update the best individual and fitness
                    if fitness < best_fitness:
                        best_individual = perturbed_individual
                        best_fitness = fitness

                # If the best fitness is better than the current best fitness, replace the individual with the new individual
                if best_fitness < fitness:
                    new_population.append(best_individual)
                    break
            else:
                # If the individual is not in the population, add it to the new population
                new_population.append(individual)

        # Replace the population with the new population
        population = new_population

        # Update the fitnesses of the population
        fitnesses = [bboo.func(individual) for individual in population]

    # Return the best individual in the final population
    return bboo.func(population[0])