import numpy as np

class ImprovedQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_scale = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            # Enhanced quantum rotation gates with adaptive theta
            theta = np.random.normal(0, 0.1, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)
            
            # Adaptive mutation step size
            self.mutation_scale = max(0.01, self.mutation_scale * 0.999)
            offspring = population + np.random.normal(0, self.mutation_scale, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution