import numpy as np

class EnhancedABCOptimizerRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.limit = int(0.6 * self.population_size)
        self.trial_limit = 100
        self.lb = -5.0
        self.ub = 5.0
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_rate = 0.5

    def __call__(self, func):
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)
        fitness_values = np.array([func(individual) for individual in population])
        
        for itr in range(self.budget):
            indexes = np.argsort(fitness_values)
            selected_solutions = population[indexes[:self.limit]]
            
            self.limit = int(0.6 * self.population_size * (1 - itr / self.budget))
            
            for i in range(self.limit):
                phi = np.random.uniform(low=-1, high=1, size=self.dim)
                mutation_strength = 1.0 / (1.0 + np.exp(-self.mutation_rate * itr / self.budget))
                new_solution = selected_solutions[i] + mutation_strength * phi * (selected_solutions[np.random.randint(self.limit)] - selected_solutions[np.random.randint(self.limit)])
                new_solution = np.clip(new_solution, self.lb, self.ub)
                new_fitness = func(new_solution)
                
                if new_fitness < fitness_values[indexes[i]]:
                    population[indexes[i]] = new_solution
                    fitness_values[indexes[i]] = new_fitness
                
            if np.min(fitness_values) < self.best_fitness:
                self.best_solution = population[np.argmin(fitness_values)]
                self.best_fitness = np.min(fitness_values)
            
        return self.best_solution