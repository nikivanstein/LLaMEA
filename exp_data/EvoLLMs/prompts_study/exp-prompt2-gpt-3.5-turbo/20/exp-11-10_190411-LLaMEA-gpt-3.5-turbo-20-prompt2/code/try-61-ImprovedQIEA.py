import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.1

    def apply_quantum_rotation(self, population):
        thetas = np.random.uniform(-np.pi, np.pi, size=(self.budget, self.dim))
        rotation_matrices = np.stack(([np.cos(thetas), -np.sin(thetas)], [np.sin(thetas), np.cos(thetas)]), axis=-1)
        return np.matmul(population, rotation_matrices)

    def apply_mutation(self, population, offspring):
        return np.where(np.random.rand(self.budget, self.dim) < 0.5,
                        population + np.random.normal(0, self.mutation_scale, size=(self.budget, self.dim)),
                        offspring)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]
        
        for _ in range(self.budget):
            population = self.apply_quantum_rotation(population)
            offspring = population + np.random.normal(0, self.mutation_scale, size=(self.budget, self.dim))
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution
        
        return best_solution