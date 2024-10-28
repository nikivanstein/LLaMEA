import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness_values = np.zeros((self.population_size, dim))
        self.best_individual = None
        self.best_fitness = -np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Refine the search space using the current population
            self.population = self.refine_search_space(self.population)
            # Evaluate the function at the current population
            fitness_values = func(self.population)
            # Update the best individual and fitness
            if self.budget == 0:
                self.best_individual = self.population[np.argmax(fitness_values)]
                self.best_fitness = np.max(fitness_values)
            else:
                self.fitness_values = fitness_values
                if np.max(fitness_values) > self.best_fitness:
                    self.best_individual = self.population[np.argmax(fitness_values)]
                    self.best_fitness = np.max(fitness_values)

    def refine_search_space(self, population):
        # Use the probability of 0.45 to refine the search space
        refined_population = np.copy(population)
        # Refine the lower bound
        refined_population[np.random.choice(population.shape[0], 10, replace=False)] += 0.5
        # Refine the upper bound
        refined_population[np.random.choice(population.shape[0], 10, replace=False)] -= 0.5
        return refined_population

# Test the algorithm
def test_black_box_optimizer():
    optimizer = BlackBoxOptimizer(budget=100, dim=10)
    func = np.linspace(-5.0, 5.0, 24)
    optimizer(func)

# Run the test
test_black_box_optimizer()