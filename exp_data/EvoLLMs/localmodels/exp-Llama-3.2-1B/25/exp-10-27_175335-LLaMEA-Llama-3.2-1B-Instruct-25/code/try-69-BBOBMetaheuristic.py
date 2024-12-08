import numpy as np
from scipy.optimize import differential_evolution

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

    def search(self, func, bounds):
        # Define the search space
        sol = None
        for _ in range(self.dim):
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

class BBOBMetaheuristic2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = None
        self.fitness_scores = []

    def __call__(self, func, bounds):
        # Define the search space
        sol = None
        for _ in range(self.population_size):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.search(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.search(func, np.random.uniform(bounds, size=self.dim)):
                # Update the solution
                sol = sol
        
        # Evaluate the fitness of each solution
        self.fitness_scores = [self.search(func, sol) for sol in sol]
        
        # Return the best solution found
        return sol

class BBOBMetaheuristicGenetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = None
        self.fitness_scores = []
        self.population Genetics = []

    def __call__(self, func, bounds):
        # Define the search space
        sol = None
        for _ in range(self.population_size):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.search(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.search(func, np.random.uniform(bounds, size=self.dim)):
                # Update the solution
                sol = sol
        
        # Evaluate the fitness of each solution
        self.fitness_scores = [self.search(func, sol) for sol in sol]
        
        # Return the best solution found
        return sol

def generate_population(func, bounds, population_size, mutation_rate):
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(bounds, size=len(bounds))
        fitness = self.evaluate_fitness(individual, func)
        population.append(individual)
        population.append(fitness)
    
    return population

def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] += np.random.uniform(-1, 1)
    
    return mutated_individual

def crossover(parent1, parent2):
    child = np.copy(parent1)
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child[i] = parent2[i]
    
    return child

def selection(population, bounds, fitness):
    selected_population = []
    for _ in range(len(population)):
        selected_index = np.random.choice(len(population), p=fitness)
        selected_population.append(population[selected_index])
    
    return selected_population

def main():
    budget = 1000
    dim = 5
    bounds = np.linspace(-5.0, 5.0, dim)
    func = lambda x: np.sin(x)
    
    population_size = 100
    mutation_rate = 0.01
    selection_probability = 0.5
    
    population = generate_population(func, bounds, population_size, mutation_rate)
    fitness_scores = [self.evaluate_fitness(individual, func) for individual in population]
    
    # Run the genetic algorithm
    genetic_algorithm = BBOBMetaGenetic(budget, dim)
    genetic_algorithm.search(func, bounds)
    
    # Run the evolutionary algorithm
    evolutionary_algorithm = BBOBMetaheuristic(budget, dim)
    evolutionary_algorithm.search(func, bounds)
    
    # Run the evolutionary algorithm with mutation
    evolutionary_algorithm2 = BBOBMetaheuristicGenetic(budget, dim)
    evolutionary_algorithm2.search(func, bounds)
    
    # Print the fitness scores
    print("Fitness scores:")
    print(f"Evolutionary Algorithm: {fitness_scores}")
    print(f"Genetic Algorithm: {fitness_scores}")
    print(f"Evolutionary Algorithm with Mutation: {fitness_scores}")

    # Print the selected solutions
    print("Selected solutions:")
    for i, individual in enumerate(population):
        print(f"Individual {i+1}: {individual}")
    print(f"Fitness scores for each individual: {fitness_scores}")

if __name__ == "__main__":
    main()