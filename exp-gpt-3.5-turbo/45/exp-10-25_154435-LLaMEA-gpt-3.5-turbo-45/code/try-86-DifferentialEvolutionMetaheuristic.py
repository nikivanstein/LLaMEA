import numpy as np

class DifferentialEvolutionMetaheuristic:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR

    def mutate(self, population, target_idx):
        candidates = [idx for idx in range(len(population)) if idx != target_idx]
        selected = np.random.choice(candidates, 3, replace=False)
        a, b, c = selected
        mutant_vector = population[a] + self.F * (population[b] - population[c])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def select(self, target_vector, trial_vector, func):
        if func(trial_vector) < func(target_vector):
            return trial_vector
        return target_vector

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            new_population = []
            for target_idx, target_vector in enumerate(population):
                mutant_vector = self.mutate(population, target_idx)
                trial_vector = self.crossover(target_vector, mutant_vector)
                selected_vector = self.select(target_vector, trial_vector, func)
                new_population.append(selected_vector)
            population = np.array(new_population)
        best_solution_idx = np.argmin([func(individual) for individual in population])
        return population[best_solution_idx]