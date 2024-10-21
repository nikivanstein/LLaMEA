# Description: MetaDifferential Evolution with Refinements for BBOB Optimization
# Code: 
import random
import numpy as np

class MetaDifferentialEvolution:
    def __init__(self, budget, dim):
        """
        Initialize the MetaDifferential Evolution algorithm.

        Parameters:
        - budget (int): The maximum number of function evaluations.
        - dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

    def __call__(self, func):
        """
        Optimize the black box function using MetaDifferential Evolution.

        Parameters:
        - func (function): The black box function to optimize.

        Returns:
        - optimized_func (function): The optimized function.
        """
        while self.budget > 0:
            # Generate a new population by perturbing the current population
            new_population = self.generate_new_population()

            # Evaluate the new population using the given budget
            new_population_evaluations = np.random.randint(1, self.budget + 1)

            # Evaluate the new population
            new_population_evaluations = np.minimum(new_population_evaluations, self.budget)

            # Select the fittest individuals from the new population
            self.population = self.select_fittest(new_population, new_population_evaluations)

            # Update the population size
            self.population_size = min(self.population_size, len(new_population))

            # Check if the population has been fully optimized
            if len(self.population) == 0:
                break

            # Perform mutation on the fittest individuals
            self.population = self.mutate(self.population)

        # Return the optimized function
        return func

    def generate_new_population(self):
        """
        Generate a new population by perturbing the current population.

        Returns:
        - new_population (numpy array): The new population.
        """
        new_population = self.population.copy()
        for _ in range(self.population_size // 2):
            # Perturb the current individual
            new_population[random.randint(0, self.dim - 1)] += random.uniform(-5.0, 5.0)

        return new_population

    def select_fittest(self, new_population, new_population_evaluations):
        """
        Select the fittest individuals from the new population.

        Parameters:
        - new_population (numpy array): The new population.
        - new_population_evaluations (numpy array): The evaluations of the new population.

        Returns:
        - fittest_population (numpy array): The fittest population.
        """
        # Calculate the fitness of each individual
        fitness = np.abs(new_population_evaluations)

        # Select the fittest individuals
        fittest_population = new_population[fitness.argsort()[:len(fitness)]]

        return fittest_population

    def mutate(self, population):
        """
        Perform mutation on the fittest individuals.

        Parameters:
        - population (numpy array): The population.

        Returns:
        - mutated_population (numpy array): The mutated population.
        """
        mutated_population = population.copy()
        for _ in range(self.mutation_rate * len(population)):
            # Select a random individual
            individual = random.choice(mutated_population)

            # Perturb the individual
            mutated_individual = individual.copy()
            for i in range(self.dim):
                # Perturb the individual
                mutated_individual[i] += random.uniform(-5.0, 5.0)

            # Clip the mutated individual to the bounds
            mutated_individual[i] = np.clip(mutated_individual[i], -5.0, 5.0)

            mutated_population[random.randint(0, len(mutated_population) - 1)] = mutated_individual

        return mutated_population

# Description: MetaDifferential Evolution with Refinements for BBOB Optimization
# Code: 
import numpy as np

class MetaDifferentialEvolutionWithRefinements:
    def __init__(self, budget, dim):
        """
        Initialize the MetaDifferential Evolution with Refinements algorithm.

        Parameters:
        - budget (int): The maximum number of function evaluations.
        - dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))

    def __call__(self, func):
        """
        Optimize the black box function using MetaDifferential Evolution with Refinements.

        Parameters:
        - func (function): The black box function to optimize.

        Returns:
        - optimized_func (function): The optimized function.
        """
        while self.budget > 0:
            # Generate a new population by perturbing the current population
            new_population = self.generate_new_population()

            # Evaluate the new population using the given budget
            new_population_evaluations = np.random.randint(1, self.budget + 1)

            # Evaluate the new population
            new_population_evaluations = np.minimum(new_population_evaluations, self.budget)

            # Select the fittest individuals from the new population
            self.population = self.select_fittest(new_population, new_population_evaluations)

            # Update the population size
            self.population_size = min(self.population_size, len(new_population))

            # Refine the mutation strategy
            self.refineMutation(self.population)

            # Check if the population has been fully optimized
            if len(self.population) == 0:
                break

            # Perform mutation on the fittest individuals
            self.population = self.mutate(self.population)

        # Return the optimized function
        return func

    def generate_new_population(self):
        """
        Generate a new population by perturbing the current population.

        Returns:
        - new_population (numpy array): The new population.
        """
        new_population = self.population.copy()
        for _ in range(self.population_size // 2):
            # Perturb the current individual
            new_population[random.randint(0, self.dim - 1)] += random.uniform(-5.0, 5.0)

        return new_population

    def select_fittest(self, new_population, new_population_evaluations):
        """
        Select the fittest individuals from the new population.

        Parameters:
        - new_population (numpy array): The new population.
        - new_population_evaluations (numpy array): The evaluations of the new population.

        Returns:
        - fittest_population (numpy array): The fittest population.
        """
        # Calculate the fitness of each individual
        fitness = np.abs(new_population_evaluations)

        # Select the fittest individuals
        fittest_population = new_population[fitness.argsort()[:len(fitness)]]

        return fittest_population

    def refineMutation(self, population):
        """
        Refine the mutation strategy.

        Parameters:
        - population (numpy array): The population.
        """
        # Select a random individual
        individual = random.choice(population)

        # Perform mutation
        mutated_individual = individual.copy()
        for i in range(self.dim):
            # Perturb the individual
            mutated_individual[i] += random.uniform(-5.0, 5.0)

        # Clip the mutated individual to the bounds
        mutated_individual[i] = np.clip(mutated_individual[i], -5.0, 5.0)

        # Replace the original individual with the mutated individual
        population[random.randint(0, len(population) - 1)] = mutated_individual

# Description: MetaDifferential Evolution with Refinements for BBOB Optimization
# Code: 
import numpy as np

# Test the MetaDifferential Evolution with Refinements algorithm
budget = 1000
dim = 10
func = lambda x: np.sin(x)
metaDifferentialEvolution = MetaDifferentialEvolutionWithRefinements(budget, dim)
optimized_func = metaDifferentialEvolution(func)

# Print the results
print("MetaDifferential Evolution with Refinements:")
print("Name:", "MetaDifferential Evolution with Refinements")
print("Description:", "MetaDifferential Evolution with Refinements for BBOB Optimization")
print("Score:", optimized_func())