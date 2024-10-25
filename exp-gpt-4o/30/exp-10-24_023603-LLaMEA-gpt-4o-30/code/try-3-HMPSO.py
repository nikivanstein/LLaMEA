import numpy as np

class HMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.memory_size = 5
        
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
        
        self.memory = {
            "c1": np.full(self.memory_size, 1.49618),
            "c2": np.full(self.memory_size, 1.49618),
            "w": np.full(self.memory_size, 0.7298)
        }
        self.memory_index = 0
        
    def update_memory(self, c1, c2, w):
        self.memory["c1"][self.memory_index] = c1
        self.memory["c2"][self.memory_index] = c2
        self.memory["w"][self.memory_index] = w
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def __call__(self, func):
        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                c1 = np.random.choice(self.memory["c1"])
                c2 = np.random.choice(self.memory["c2"])
                w = np.random.choice(self.memory["w"])

                self.velocity[i] = (w * self.velocity[i] +
                                    c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                    c2 * r2 * (self.global_best - self.population[i]))
                
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                trial_fitness = func(self.population[i])
                eval_count += 1

                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best[i] = self.population[i]
                    self.personal_best_fitness[i] = trial_fitness
                    
                if trial_fitness < self.global_best_fitness:
                    self.global_best = self.population[i]
                    self.global_best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            self.update_memory(c1, c2, w)
        
        return self.global_best, self.global_best_fitness