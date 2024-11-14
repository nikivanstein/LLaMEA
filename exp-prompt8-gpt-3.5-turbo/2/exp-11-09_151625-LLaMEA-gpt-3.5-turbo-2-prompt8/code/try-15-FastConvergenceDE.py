class FastConvergenceDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.F = 0.5
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutation(self, target, population, fitness):
        radius = np.clip(np.mean(fitness), 0.1, 1.0)
        step_size = np.clip(np.std(fitness), 0.1, 1.0)
        mutant = target + step_size * self.F * (population[np.random.randint(self.pop_size)] - target)
        return target + radius * mutant

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        fitness = [func(ind) for ind in population]

        for _ in range(self.budget):
            new_population = []
            for target in population:
                a, b, c = population[np.random.choice(range(self.pop_size), 3, replace=False)]
                trial = self.mutation(target, population, fitness)
                mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(mask, trial, target)
                new_population.append(offspring)

            population = np.array(new_population)
            best_solution = population[np.argmin([func(ind) for ind in population])]
            fitness = [func(ind) for ind in population]

        return best_solution