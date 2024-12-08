import numpy as np

class EnhancedCrowdingABCOptimizerEnhanced:
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
        self.initial_mutation_rate = 0.5  # Initial mutation rate
        self.mutation_rate = self.initial_mutation_rate

    def __call__(self, func):
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)
        fitness_values = np.array([func(individual) for individual in population])
        
        for itr in range(self.budget):
            indexes = np.argsort(fitness_values)
            selected_solutions = population[indexes[:self.limit]]
            
            # Introduce crowding-based selection to enhance diversity
            crowding_indexes = np.argsort(-np.linalg.norm(selected_solutions - np.mean(selected_solutions, axis=0), axis=1))
            selected_solutions = selected_solutions[crowding_indexes]
            
            self.limit = int(0.6 * self.population_size * (1 - itr / self.budget))
            
            for i in range(self.limit):
                phi = np.random.uniform(low=-1, high=1, size=self.dim)
                chaos_sequence = np.tanh(np.sin(phi) + np.mean(fitness_values) * self.mutation_rate)  # Dynamic chaos sequence generation
                mutation_strength = 1.0 / (1.0 + np.exp(-self.mutation_rate * (fitness_values[indexes[i]] - np.min(fitness_values))))
                
                if np.random.rand() < 0.1:  # 10% chance to perform Levy flights
                    levy = np.random.standard_cauchy(self.dim) / np.sqrt(np.abs(np.random.normal(0, 1, self.dim)))
                    new_solution = selected_solutions[i] + 0.01 * levy
                else:
                    if np.random.rand() < 0.5:  # Differential evolution strategy with a 50% probability
                        r1, r2 = np.random.randint(0, self.limit), np.random.randint(0, self.limit)
                        diff = selected_solutions[r1] - selected_solutions[r2]
                        new_solution = selected_solutions[i] + mutation_strength * diff
                    else:
                        new_solution = selected_solutions[i] + mutation_strength * chaos_sequence * (selected_solutions[np.random.randint(self.limit)] - selected_solutions[np.random.randint(self.limit)])
                        # Include an additional exploration strategy based on enhanced chaotic map for mutation
                        chaotic_map = np.sin(np.log(np.abs(chaos_sequence + 1.0)))  # Novel chaotic map
                        new_solution += 0.1 * chaotic_map * np.random.randn(self.dim) * np.abs(np.tanh(chaos_sequence))
                    
                new_solution = np.clip(new_solution, self.lb, self.ub)
                new_fitness = func(new_solution)
                
                if new_fitness < fitness_values[indexes[i]]:
                    population[indexes[i]] = new_solution
                    fitness_values[indexes[i]] = new_fitness
                
            if np.min(fitness_values) < self.best_fitness:
                self.best_solution = population[np.argmin(fitness_values)]
                self.best_fitness = np.min(fitness_values)
                
                # Dynamic mutation rate adjustment based on fitness improvement
                diversity = np.mean(np.std(selected_solutions, axis=0))
                self.mutation_rate = self.initial_mutation_rate + 0.1 * diversity
                
        return self.best_solution