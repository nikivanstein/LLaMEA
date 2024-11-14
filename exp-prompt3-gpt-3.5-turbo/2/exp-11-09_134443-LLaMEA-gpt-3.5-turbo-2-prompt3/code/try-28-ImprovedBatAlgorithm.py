import numpy as np

class ImprovedBatAlgorithm(BatAlgorithm):
    def __call__(self, func):
        best_solution = self.population[0]
        best_fitness = func(best_solution)
        for _ in range(self.budget):
            frequencies = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.population_size)
            self.v += (self.population - func(self.population)) * frequencies[:, None]
            self.population = np.clip(self.population + self.v, self.v_min, self.v_max)
            for i in range(self.population_size):
                if np.random.rand() > self.Q[i]:
                    self.population[i] = func(np.random.uniform(-5.0, 5.0, self.dim))
            self.Q = self.Q_min + (self.Q_max - self.Q_min) * np.random.rand(self.population_size)
            self.A_min = self.alpha * self.A_min
            self.A = self.A_min + (self.A_max - self.A_min) * np.random.rand(self.population_size)
            self.f_min = self.f_min + self.gamma
            self.f_max = self.f_max * self.alpha
            
            current_fitness = func(self.population)
            for i in range(self.population_size):
                if current_fitness[i] < best_fitness:
                    best_solution = self.population[i]
                    best_fitness = current_fitness[i]
        
        return best_solution