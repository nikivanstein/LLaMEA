import numpy as np

class MetaheuristicOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            population = self.de_update(population, func)
        best_solution = min(population, key=lambda x: func(x['position']))
        return best_solution['position']

    def initialize_population(self):
        return [{'position': np.random.uniform(-5.0, 5.0, self.dim)} for _ in range(self.population_size)]

    def de_update(self, population, func):
        for i in range(self.population_size):
            x, a, b, c = population[i]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position']
            if np.random.rand() < 0.2:
                self.f = np.clip(np.random.normal(self.f, 0.1), self.min_f, self.max_f)  # Probabilistic mutation rate change
                individual_line = np.random.randint(self.dim)
                x[individual_line] = np.clip(np.random.uniform(-5.0, 5.0), -5.0, 5.0)  # Individual line change
            mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) <= self.cr, mutant, x)
            if func(trial) < func(x):
                population[i]['position'] = trial.copy()
        return population