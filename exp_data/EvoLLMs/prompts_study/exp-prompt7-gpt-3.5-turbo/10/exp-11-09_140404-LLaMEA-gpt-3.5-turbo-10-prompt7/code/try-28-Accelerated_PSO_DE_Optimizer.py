import numpy as np

class Accelerated_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            # Enhanced DE mutation strategy
            for i in range(len(population)):
                candidate = population[i]
                mutant1 = population[np.random.choice(len(population))]
                mutant2 = population[np.random.choice(len(population))]
                mutant3 = population[np.random.choice(len(population))]
                
                # Updated mutation process
                crossover_point = np.random.randint(self.dim)
                mutant3[crossover_point + 1:] = mutant1[crossover_point + 1:] + 0.5 * (mutant2[crossover_point + 1:] - mutant1[crossover_point + 1:])
                
                trial = mutant3 if func(mutant3) < func(candidate) else candidate
                population[i] = trial
        
        population = initialize_population(50)
        while self.budget > 0:
            # Improved PSO update mechanism
            global_best = population[np.argmin([func(individual) for individual in population])]
            for i in range(len(population)):
                particle = population[i]
                velocity = np.random.uniform(-1, 1, size=self.dim) * (particle - global_best)
                population[i] = particle + velocity
                population[i] = np.clip(population[i], -5.0, 5.0)
            optimize_population(population)
            self.budget -= 1
        
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution