import numpy as np
import random

class MultilevelGeneticAlgorithm:
    def __init__(self, budget, dim, levels):
        """
        Initialize the multilevel genetic algorithm.

        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            levels (int): The number of levels in the multilevel optimization.
        """
        self.budget = budget
        self.dim = dim
        self.levels = levels
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population with random individuals.

        Returns:
            list: A list of individuals, each represented as a dictionary with the following keys:
                - 'fitness': The fitness of the individual.
                -'strategy': The strategy used to optimize the function.
        """
        population = []
        for _ in range(self.population_size):
            individual = {}
            individual['fitness'] = -np.inf
            individual['strategy'] = 'random'
            for _ in range(self.levels):
                individual['strategy'] = 'level' + str(_)
                individual['fitness'] = random.uniform(0, 1)
            population.append(individual)
        return population

    def mutate(self, individual):
        """
        Mutate an individual by changing its strategy.

        Args:
            individual (dict): The individual to mutate.
        """
        individual['strategy'] = random.choice(['random', 'level' + str(random.randint(0, self.levels - 1))])

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child.

        Args:
            parent1 (dict): The first parent.
            parent2 (dict): The second parent.

        Returns:
            dict: The child.
        """
        child = {}
        for key in ['fitness','strategy']:
            child[key] = (parent1[key] + parent2[key]) / 2
        for key in ['fitness','strategy']:
            if key =='strategy':
                if random.random() < 0.5:
                    child[key] = random.choice([parent1[key], parent2[key]])
                else:
                    child[key] = random.choice([parent1[key] + parent2[key] for _ in range(2)])
            else:
                child[key] = (parent1[key] + parent2[key]) / 2
        return child

    def optimize(self, individual):
        """
        Optimize an individual using the specified strategy.

        Args:
            individual (dict): The individual to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Select the strategy based on the individual's fitness
        strategy = individual['strategy']
        if strategy == 'random':
            return random.uniform(0, 1)
        elif strategy == 'level' + str(random.randint(0, self.levels - 1)):
            return random.uniform(0, 1)
        elif strategy == 'level' + str(random.randint(0, self.levels - 1)):
            return random.uniform(0, 1)
        else:
            raise ValueError("Invalid strategy")

    def __call__(self, func):
        """
        Optimize the black box function using the multilevel genetic algorithm.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population
        population = self.initialize_population()

        # Run the genetic algorithm
        while len(population) > 0 and self.budget > 0:
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x['fitness'], reverse=True)[:self.population_size // 2]
            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = random.choice(fittest_individuals)
                parent2 = random.choice(fittest_individuals)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            # Replace the old population with the new population
            population = new_population

        # Return the fittest individual
        return max(population, key=lambda x: x['fitness'])

# Example usage:
if __name__ == "__main__":
    algorithm = MultilevelGeneticAlgorithm(budget=100, dim=10, levels=3)
    func = lambda x: x**2
    optimized_value = algorithm(__call__(func))
    print(f"Optimized value: {optimized_value}")