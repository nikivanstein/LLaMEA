import numpy as np

class AQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def quantum_crossover(self, parents):
        alpha = np.random.uniform(0, 1, self.dim)
        child_1 = alpha * parents[0] + (1 - alpha) * parents[1]
        child_2 = (1 - alpha) * parents[0] + alpha * parents[1]
        return np.clip(child_1, self.lower_bound, self.upper_bound), np.clip(child_2, self.lower_bound, self.upper_bound)

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        while self.evaluations < self.budget:
            fitness = np.array([self.evaluate(func, ind) for ind in self.population])
            if np.min(fitness) < self.best_fitness:
                self.best_fitness = np.min(fitness)
                self.best_solution = self.population[np.argmin(fitness)]

            selected_indexes = np.random.choice(self.pop_size, self.pop_size, replace=True)
            selected_parents = self.population[selected_indexes]
            new_population = []
            
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_parents[i], selected_parents[i+1]
                offspring_1, offspring_2 = self.quantum_crossover((p1, p2))
                new_population.extend([offspring_1, offspring_2])

            self.population = np.array(new_population)
        
        return self.best_solution