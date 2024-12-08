import numpy as np
import random
import operator

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer.

        Parameters:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                x = x[:-1] + np.random.uniform(-0.1, 0.1, self.dim)
        
        # Return the optimized value of the function
        return x[-1]

    def genetic_algorithm(self, func, initial_population, mutation_rate, elite_size):
        """
        Implement a Genetic Algorithm to optimize the black box function.

        Parameters:
        func (function): The black box function to optimize.
        initial_population (list): The initial population of individuals.
        mutation_rate (float): The probability of mutation.
        elite_size (int): The number of individuals in the elite population.

        Returns:
        tuple: The optimized individual and the fitness score.
        """
        # Initialize the population with random individuals
        population = initial_population
        
        # Initialize the elite population
        elite = population[:elite_size]
        
        # Evaluate the fitness of the initial population
        fitness = [self.__call__(func, individual) for individual in population]
        
        # Perform crossover and mutation
        for _ in range(100):
            # Select parents using tournament selection
            parents = self.tournament_selection(population, fitness)
            
            # Perform crossover
            children = []
            for _ in range(len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                children.append(child)
            
            # Perform mutation
            for child in children:
                if random.random() < mutation_rate:
                    child = self.mutation(child)
            
            # Evaluate the fitness of the new population
            new_fitness = [self.__call__(func, individual) for individual in children]
            
            # Replace the old population with the new one
            population = children
            fitness = new_fitness
        
        # Return the elite individual and its fitness score
        return elite[0], fitness[elite_size - 1]

    def tournament_selection(self, population, fitness):
        """
        Select parents using tournament selection.

        Parameters:
        population (list): The population of individuals.
        fitness (list): The fitness scores of the individuals.

        Returns:
        list: The selected parents.
        """
        selected_parents = []
        for _ in range(len(population)):
            # Randomly select an individual
            individual = random.choice(population)
            
            # Get the fitness score of the individual
            fitness_score = fitness[population.index(individual)]
            
            # Get the top k individuals with the highest fitness score
            top_k = random.sample(population, k=1)[0]
            
            # Check if the individual is better than the top k individuals
            if fitness_score > top_k[1]:
                selected_parents.append(individual)
        
        return selected_parents

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Parameters:
        parent1 (individual): The first parent.
        parent2 (individual): The second parent.

        Returns:
        individual: The child individual.
        """
        # Get the crossover point
        crossover_point = random.randint(1, self.dim - 1)
        
        # Perform crossover
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child

    def mutation(self, individual):
        """
        Perform mutation on an individual.

        Parameters:
        individual (individual): The individual to mutate.

        Returns:
        individual: The mutated individual.
        """
        # Get the mutation point
        mutation_point = random.randint(0, self.dim - 1)
        
        # Perform mutation
        individual[mutation_point] += np.random.uniform(-1, 1)
        
        return individual

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iterative Refining of the Search Space using Stochastic Gradient Descent and Genetic Algorithm"

# Code
def stgd(x, func, epsilon, learning_rate):
    """
    Iteratively refine the search space using Stochastic Gradient Descent.

    Parameters:
    x (numpy array): The current point in the search space.
    func (function): The black box function to optimize.
    epsilon (float): The step size for the gradient descent update.
    learning_rate (float): The step size for the gradient descent update.

    Returns:
    numpy array: The updated point in the search space.
    """
    y = func(x)
    grad = (y - x[-1]) / epsilon
    x = x[:-1] + np.random.uniform(-epsilon, epsilon, self.dim)
    return x

def func(x):
    return x**2

def genetic_algorithm(func, initial_population, mutation_rate, elite_size):
    optimizer = BlackBoxOptimizer(1000, 10)
    optimized_x = optimizer(func, 0, 1)
    print(optimized_x)

    # Initialize the population with random individuals
    population = initial_population
    
    # Initialize the elite population
    elite = population[:elite_size]
    
    # Evaluate the fitness of the initial population
    fitness = [optimizer.__call__(func, individual) for individual in population]
    
    # Perform crossover and mutation
    for _ in range(100):
        # Select parents using tournament selection
        parents = optimizer.tournament_selection(population, fitness)
        
        # Perform crossover
        children = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = optimizer.crossover(parent1, parent2)
            children.append(child)
        
        # Perform mutation
        for child in children:
            if random.random() < mutation_rate:
                child = optimizer.mutation(child)
        
        # Evaluate the fitness of the new population
        new_fitness = [optimizer.__call__(func, individual) for individual in children]
        
        # Replace the old population with the new one
        population = children
        fitness = new_fitness
    
    # Return the elite individual and its fitness score
    return elite[0], fitness[elite_size - 1]

# Example usage:
genetic_algorithm(func, np.random.uniform(-10, 10, 100), 0.1, 10)