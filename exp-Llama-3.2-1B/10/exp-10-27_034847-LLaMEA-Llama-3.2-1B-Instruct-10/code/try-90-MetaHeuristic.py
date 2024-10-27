import numpy as np
import random

class MetaHeuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of local search and evolutionary strategies to find the optimal solution.
    """
    def __init__(self, budget, dim):
        """
        Initializes the MetaHeuristic algorithm.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.bounds = None

    def __call__(self, func):
        """
        Optimizes the black box function using MetaHeuristic.
        
        Args:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        if self.func is None:
            raise ValueError("The black box function must be initialized before calling this method.")
        
        # Initialize the search space
        self.search_space = [self.bounds] * self.dim
        self.bounds = [(-5.0, 5.0)] * self.dim
        
        # Initialize the optimal solution and its cost
        opt_solution = None
        opt_cost = float('inf')
        
        # Perform local search using evolutionary strategies
        for _ in range(self.budget):
            # Initialize the population
            population = [self.perturb(self.search_space, self.bounds) for _ in range(100)]
            
            # Evaluate the population using the black box function
            fitness = [self.func(individual) for individual in population]
            
            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitness, population), reverse=True)[:self.budget//2]]
            
            # Crossover and mutation
            children = []
            for i in range(0, len(fittest_individuals), 2):
                parent1, parent2 = fittest_individuals[i], fittest_individuals[i+1]
                child1 = self.perturb(self.search_space, self.bounds)
                child2 = self.perturb(self.search_space, self.bounds)
                child1, child2 = self.crossover(child1, child2)
                child1, child2 = self.mutation(child1, child2)
                children.extend([child1, child2])
            
            # Update the optimal solution and its cost
            if fitness.index(min(fitness)) < self.budget//2:
                opt_solution = fittest_individuals[0]
                opt_cost = min(fitness)
        
        # Return the optimal solution and its cost
        return opt_solution, opt_cost

    def perturb(self, search_space, bounds):
        """
        Generates a new solution by perturbing the current solution.
        
        Args:
        search_space (list): The current search space.
        bounds (list): The current bounds of the search space.
        
        Returns:
        list: A new solution generated by perturbing the current solution.
        """
        # Generate a new solution by randomly perturbing the current solution
        new_solution = [bounds[0] + np.random.uniform(-1, 1) * (bounds[1] - bounds[0]) for _ in range(self.dim)]
        
        # Ensure the new solution is within the bounds
        new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
        return new_solution

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent individuals.
        
        Args:
        parent1 (list): The first parent individual.
        parent2 (list): The second parent individual.
        
        Returns:
        list: The child individual generated by crossover.
        """
        # Select a random crossover point
        crossover_point = random.randint(0, len(parent1) - 1)
        
        # Create the child individual
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child

    def mutation(self, individual, mutation_rate):
        """
        Applies mutation to an individual.
        
        Args:
        individual (list): The individual to mutate.
        mutation_rate (float): The probability of mutation.
        
        Returns:
        list: The mutated individual.
        """
        # Apply mutation with probability mutation_rate
        if random.random() < mutation_rate:
            # Generate a random index for mutation
            index = random.randint(0, len(individual) - 1)
            
            # Swap the individual with the randomly selected index
            individual[index], individual[index + 1] = individual[index + 1], individual[index]
        
        return individual

# Description: Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
# import numpy as np
# import random

# class MetaHeuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.search_space = None
#         self.bounds = None

#     def __call__(self, func):
#         if self.func is None:
#             raise ValueError("The black box function must be initialized before calling this method.")
#         # Initialize the search space
#         self.search_space = [self.bounds] * self.dim
#         self.bounds = [(-5.0, 5.0)] * self.dim
        
#         # Initialize the optimal solution and its cost
#         opt_solution = None
#         opt_cost = float('inf')
        
#         # Perform local search using evolutionary strategies
#         for _ in range(self.budget):
#             # Initialize the population
#             population = [self.perturb(self.search_space, self.bounds) for _ in range(100)]
            
#             # Evaluate the population using the black box function
#             fitness = [self.func(individual) for individual in population]
            
#             # Select the fittest individuals
#             fittest_individuals = [individual for _, individual in sorted(zip(fitness, population), reverse=True)[:self.budget//2]]
            
#             # Crossover and mutation
#             children = []
#             for i in range(0, len(fittest_individuals), 2):
#                 parent1, parent2 = fittest_individuals[i], fittest_individuals[i+1]
#                 child1 = self.perturb(self.search_space, self.bounds)
#                 child2 = self.perturb(self.search_space, self.bounds)
#                 child1, child2 = self.crossover(child1, child2)
#                 child1, child2 = self.mutation(child1, child2)
#                 children.extend([child1, child2])
            
#             # Update the optimal solution and its cost
#             if fitness.index(min(fitness)) < self.budget//2:
#                 opt_solution = fittest_individuals[0]
#                 opt_cost = min(fitness)
        
#     def perturb(self, search_space, bounds):
#         # Generate a new solution by randomly perturbing the current solution
#         new_solution = [bounds[0] + np.random.uniform(-1, 1) * (bounds[1] - bounds[0]) for _ in range(self.dim)]
        
#         # Ensure the new solution is within the bounds
#         new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
#         return new_solution

#     def crossover(self, parent1, parent2):
#         # Select a random crossover point
#         crossover_point = random.randint(0, len(parent1) - 1)
        
#         # Create the child individual
#         child = parent1[:crossover_point] + parent2[crossover_point:]
        
#         return child

#     def mutation(self, individual, mutation_rate):
#         # Apply mutation with probability mutation_rate
#         if random.random() < mutation_rate:
#             # Generate a random index for mutation
#             index = random.randint(0, len(individual) - 1)
            
#             # Swap the individual with the randomly selected index
#             individual[index], individual[index + 1] = individual[index + 1], individual[index]
        
#         return individual