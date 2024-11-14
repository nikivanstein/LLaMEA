import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.initial_mutation_factor = 0.6
        self.initial_cross_prob = 0.85
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.best_value = np.min(fitness)
        self.best_solution = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size
        mutation_factor = self.initial_mutation_factor
        cross_prob = self.initial_cross_prob

        while self.evaluations < self.budget:
            next_generation = []

            for i in range(self.population_size):
                variant_choice = np.random.rand()
                if variant_choice < 0.3:
                    next_gen_candidate = self.de_variant_1(i, mutation_factor, cross_prob)
                elif variant_choice < 0.6:
                    next_gen_candidate = self.de_variant_2(i, mutation_factor, cross_prob)
                else:
                    next_gen_candidate = self.de_variant_3(i, mutation_factor, cross_prob)
                
                next_generation.append(next_gen_candidate)

            self.population = np.array(next_generation)
            fitness = np.apply_along_axis(func, 1, self.population)
            self.evaluations += self.population_size

            current_best_value = np.min(fitness)
            if current_best_value < self.best_value:
                self.best_value = current_best_value
                self.best_solution = self.population[np.argmin(fitness)]

            mutation_factor = 0.8 * mutation_factor + 0.2 * np.random.rand()
            cross_prob = 0.8 * cross_prob + 0.2 * np.random.rand()

        return self.best_solution

    def de_variant_1(self, index, mutation_factor, cross_prob):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        return self.crossover(self.population[index], mutant, cross_prob)

    def de_variant_2(self, index, mutation_factor, cross_prob):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + mutation_factor * (c - b), self.lower_bound, self.upper_bound)
        return self.crossover(self.population[index], mutant, cross_prob)
    
    def de_variant_3(self, index, mutation_factor, cross_prob):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c, d = self.population[np.random.choice(idxs, 4, replace=False)]
        mutant = np.clip(a + mutation_factor * (b - c + d), self.lower_bound, self.upper_bound)
        return self.crossover(self.population[index], mutant, cross_prob)

    def crossover(self, target, mutant, cross_prob):
        cross_points = np.random.rand(self.dim) < cross_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial