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
        self.population = self.initialize_population()
        self.population_strategies = []
        self.population_fitness = []
        
    def initialize_population(self):
        """
        Initialize the population with random solutions.
        
        Returns:
            list: The initialized population.
        """
        
        # Initialize the population with random solutions
        population = [(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)) for _ in range(100)]
        
        # Randomly select a strategy for each individual
        for individual in population:
            strategy = random.choice(['random', 'bounded', 'perturbation'])
            self.population_strategies.append(strategy)
            self.population_fitness.append(individual[1])
        
        return population
    
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
    
    def mutate(self, individual):
        """
        Mutate an individual using a specified strategy.
        
        Args:
            individual (tuple): The individual to mutate.
        
        Returns:
            tuple: The mutated individual.
        """
        
        # Randomly select a strategy
        strategy = random.choice(self.population_strategies)
        
        # Mutate the individual using the selected strategy
        if strategy == 'random':
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)
            # Mutate the individual at the selected point
            individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
        elif strategy == 'bounded':
            # Generate a random bound
            bound = random.uniform(-5.0, 5.0)
            # Mutate the individual within the bound
            individual = (individual[0] + random.uniform(-bound, bound), individual[1] + random.uniform(-bound, bound))
        elif strategy == 'perturbation':
            # Generate a random perturbation
            perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
            # Mutate the individual with the perturbation
            individual = (individual[0] + perturbation[0], individual[1] + perturbation[1])
        
        return individual
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = func(individual)
        
        # Update the population fitness
        self.population_fitness.append(fitness)
        
        return fitness
    
    def update_population(self, num_iterations):
        """
        Update the population using the specified number of iterations.
        
        Args:
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The updated population.
        """
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        # Mutate the population
        for individual in self.population:
            self.population_strategies.append(random.choice(['random', 'bounded', 'perturbation']))
            self.population_fitness.append(self.evaluate_fitness(individual))
        
        return self.population
    
    def __str__(self):
        """
        Return a string representation of the optimizer.
        
        Returns:
            str: A string representation of the optimizer.
        """
        
        # Return a string representation of the optimizer
        return 'Randomized Black Box Optimization Algorithm with Evolutionary Strategies'