import numpy as np

class SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.memory_size = 5
        self.cross_prob = 0.9
        self.F = 0.5
        
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.memory = {
            "F": np.full(self.memory_size, self.F),
            "CR": np.full(self.memory_size, self.cross_prob)
        }
        self.memory_index = 0
        self.success_rates = np.zeros(self.memory_size)
        
    def update_memory(self, F, CR, success):
        self.memory["F"][self.memory_index] = F
        self.memory["CR"][self.memory_index] = CR
        self.success_rates[self.memory_index] = success
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def select_parameters(self):
        probabilities = self.success_rates / np.sum(self.success_rates)
        if np.sum(probabilities) > 0:
            index = np.random.choice(self.memory_size, p=probabilities)
        else:
            index = np.random.randint(self.memory_size)
        return self.memory["F"][index], self.memory["CR"][index]

    def __call__(self, func):
        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                F, CR = self.select_parameters()
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if eval_count >= self.budget:
                    break

                if trial_fitness < self.fitness[i]:
                    success = 1.0
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                else:
                    success = 0.0
                
                self.update_memory(F, CR, success)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]