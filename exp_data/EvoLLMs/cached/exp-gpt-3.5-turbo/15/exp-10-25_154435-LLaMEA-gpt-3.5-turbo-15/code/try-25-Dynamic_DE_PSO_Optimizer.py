import numpy as np

class Dynamic_DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, c1=2.0, c2=2.0, w=0.7, f=0.5, cr=0.9, f_min=0.2, f_max=0.8, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f = f
        self.cr = cr
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max

    def __call__(self, func):
        def evaluate_population(population):
            return np.array([func(ind) for ind in population])

        def clip_bounds(population):
            return np.clip(population, -5.0, 5.0)

        def adapt_parameters(iteration):
            nonlocal self.f, self.cr
            self.f = self.f_max - (self.f_max - self.f_min) * iteration / self.budget
            self.cr = self.cr_max - (self.cr_max - self.cr_min) * iteration / self.budget

        def mutate(population, target_idx):
            candidates = list(range(len(population)))
            candidates.remove(target_idx)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            f = np.random.normal(self.f, 0.1)  # Adaptive scaling of differential weight
            mutant = population[a] + f * (population[b] - population[c])
            return clip_bounds(mutant)

        def crossover(target, mutant):
            trial = np.copy(target)
            crossover_points = np.random.rand(self.dim) < self.cr
            if not np.any(crossover_points):
                crossover_points[np.random.randint(0, self.dim)] = True
            trial[crossover_points] = mutant[crossover_points]
            return trial

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for it in range(self.budget - self.pop_size):
            adapt_parameters(it)
            for i in range(self.pop_size):
                mutant = mutate(population, i)
                trial = crossover(population[i], mutant)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
        return best_solution