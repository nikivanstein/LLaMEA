class DynamicPopSizeDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 # Updated to allow dynamic adjustment
        self.F = 0.5
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.min_pop_size = 5
        self.max_pop_size = 20

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        diversity = np.mean([np.linalg.norm(ind - best_solution) for ind in population])
        fitness = [func(ind) for ind in population]

        for _ in range(self.budget):
            new_population = []
            for target in population:
                a, b, c = population[np.random.choice(range(self.pop_size), 3, replace=False)]
                trial = self.mutation(target, population, diversity, fitness)
                mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(mask, trial, target)
                new_population.append(offspring)

            population = np.array(new_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]
            new_diversity = np.mean([np.linalg.norm(ind - best_solution) for ind in population])
            diversity = max(0.9 * diversity + 0.1 * new_diversity, 1e-6)
            fitness = [func(ind) for ind in population]

            self.pop_size = min(self.max_pop_size, max(self.min_pop_size, int(np.mean([func(ind) for ind in population]) * 10)))

        return best_solution