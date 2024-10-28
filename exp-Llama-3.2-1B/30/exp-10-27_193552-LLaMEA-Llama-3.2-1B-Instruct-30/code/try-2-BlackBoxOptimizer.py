import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(solution)
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbed_solution
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

    def mutate(self, solution):
        """
        Mutate the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The mutated solution.
        """
        
        # Generate a random mutation in the search space
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the mutation
        solution = (solution[0] + mutation[0], solution[1] + mutation[1])
        
        return solution

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1 (tuple): The first parent.
            parent2 (tuple): The second parent.
        
        Returns:
            tuple: The child.
        """
        
        # Generate a random crossover point
        crossover_point = random.randint(0, self.dim - 1)
        
        # Create the child
        child = (parent1[:crossover_point] + parent2[crossover_point:], parent1[crossover_point + 1:])
        
        return child

# Description: Black Box Optimization using Genetic Algorithm with Evolutionary Strategies
# Code: 
# ```python
def fitness_func(solution):
    # Evaluate the fitness of the solution
    return 1 / (solution[0] + solution[1])

def black_box_optimizer(budget, dim):
    """
    Black Box Optimization using Genetic Algorithm with Evolutionary Strategies.
    
    Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
    
    Returns:
        tuple: The optimal solution and its fitness.
    """
    
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget, dim)
    
    # Initialize the population
    population = [optimizer.run(fitness_func, 1000) for _ in range(10)]
    
    # Evolve the population
    for _ in range(100):
        # Select the fittest individuals
        fittest_individuals = population[np.argsort(population, axis=0)]
        
        # Create a new population
        new_population = []
        
        # Perform crossover and mutation
        for _ in range(10):
            # Select two parents
            parent1, parent2 = fittest_individuals[np.random.choice(fittest_individuals.shape[0], 2, replace=False)]
            
            # Perform crossover and mutation
            child = optimizer.crossover(parent1, parent2)
            child = optimizer.mutate(child)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Replace the old population with the new population
        population = new_population
    
    # Return the fittest individual
    return population[np.argmax(population, axis=0)]

# Run the optimizer
optimizer = BlackBoxOptimizer(1000, 10)
optimal_solution, optimal_fitness = black_box_optimizer(1000, 10)
print("Optimal solution:", optimal_solution)
print("Optimal fitness:", optimal_fitness)