import numpy as np
import random
import copy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

class GeneticProgrammingBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = [copy.deepcopy(self.search(func)) for _ in range(self.population_size)]
        self.population_fitness = [self.search(func) for func in self.population]

    def mutate(self, individual):
        # Randomly mutate the individual
        index1, index2 = random.sample(range(len(individual)), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
        
        # Check if the mutation is within the bounds
        if individual[index1] < -5.0:
            individual[index1] = -5.0
        elif individual[index1] > 5.0:
            individual[index1] = 5.0
        
        if individual[index2] < -5.0:
            individual[index2] = -5.0
        elif individual[index2] > 5.0:
            individual[index2] = 5.0
        
        # Calculate the fitness of the mutated individual
        fitness = self.search(func)
        
        # Return the mutated individual and its fitness
        return individual, fitness

    def crossover(self, parent1, parent2):
        # Perform crossover between the two parents
        child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
        
        # Check if the child is within the bounds
        if child[-1] < -5.0:
            child[-1] = -5.0
        elif child[-1] > 5.0:
            child[-1] = 5.0
        
        # Calculate the fitness of the child
        fitness = self.search(func)
        
        # Return the child and its fitness
        return child, fitness

    def selection(self):
        # Select the fittest individuals
        sorted_indices = sorted(range(self.population_size), key=self.population_fitness)
        self.population = [self.population[i] for i in sorted_indices]
        self.population_fitness = [self.population_fitness[i] for i in sorted_indices]

# Select the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic = GeneticProgrammingBBOBMetaheuristic(budget=1000, dim=5)

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.mutate(individual) for individual in genetic_programming_bbbometaheuristic.population]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.crossover(parent1, parent2) for parent1, parent2 in zip(genetic_programming_bbbometaheuristic.population, genetic_programming_bbbometaheuristic.population)]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.mutate(individual) for individual in genetic_programming_bbbometaheuristic.population]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.crossover(parent1, parent2) for parent1, parent2 in zip(genetic_programming_bbbometaheuristic.population, genetic_programming_bbbometaheuristic.population)]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.mutate(individual) for individual in genetic_programming_bbbometaheuristic.population]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")

# Update the GeneticProgrammingBBOBMetaheuristic
genetic_programming_bbbometaheuristic.population = [genetic_programming_bbbometaheuristic.crossover(parent1, parent2) for parent1, parent2 in zip(genetic_programming_bbbometaheuristic.population, genetic_programming_bbbometaheuristic.population)]

# Evaluate the function 1000 times
for _ in range(1000):
    func = random.choice(list(genetic_programming_bbbometaheuristic.population[0]))
    func_evals = genetic_programming_bbbometaheuristic.search(func)
    print(f"func_evals: {func_evals}")