import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluated_solutions = []
        self.fitness_values = []

    def __call__(self, func):
        evaluations = 0
        rf_model = RandomForestRegressor()
        
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                if len(self.evaluated_solutions) >= 5:
                    rf_model.fit(self.evaluated_solutions, self.fitness_values)
                    predicted_fitness = rf_model.predict([trial])[0]
                    if predicted_fitness < self.best_fitness:
                        trial_fitness = func(trial)
                    else:
                        trial_fitness = func(self.population[i])
                else:
                    trial_fitness = func(trial)
                
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

                self.evaluated_solutions.append(trial)
                self.fitness_values.append(trial_fitness)

            self.population = new_population

        return self.best_solution, self.best_fitness