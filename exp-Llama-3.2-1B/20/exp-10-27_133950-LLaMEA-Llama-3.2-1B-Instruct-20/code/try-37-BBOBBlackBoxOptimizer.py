import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        """Generate the initial population of individuals."""
        return [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        """Generate a new individual within the search space."""
        return tuple(np.random.uniform(self.search_space) for _ in range(self.dim))

    def evaluate_fitness(self, individual):
        """Evaluate the fitness of an individual using the given function."""
        func = lambda x: x**2
        return func(individual)

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def mutate(self, individual):
        """Mutate an individual by changing one of its values."""
        index = np.random.randint(0, self.dim)
        new_value = np.random.uniform(-5.0, 5.0)
        individual[index] = new_value
        return individual

    def crossover(self, parent1, parent2):
        """Crossover two parents to produce a child."""
        child = tuple(np.random.uniform(self.search_space) for _ in range(self.dim))
        child[:index1] = parent1[:index1]
        child[index1:index2] = parent2[:index2]
        child[index2:] = parent2[index2:]
        return child

    def selection(self, population):
        """Select the fittest individuals to reproduce."""
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        sorted_indices = np.argsort(fitnesses)
        return [population[i] for i in sorted_indices[:self.population_size//2]]

    def run(self, max_generations):
        """Run the evolutionary algorithm."""
        population = self.population
        for generation in range(max_generations):
            new_population = self.selection(population)
            population = self.population + new_population
            population = self.generate_population()
            individuals = self.evaluate_fitness(population)
            best_individual = max(individuals, key=individuals.get)
            if self.evaluate_fitness(best_individual) > individuals[best_individual]:
                best_individual = individuals[best_individual]
                population = self.population + [best_individual]
        return best_individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
best_individual = optimizer.run(100)
print(best_individual)