import numpy as np

class Adaptive_DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, c1=2.0, c2=2.0, w=0.7, f=0.5, cr=0.9, p=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.f = f
        self.cr = cr
        self.p = p

    def __call__(self, func):
        def evaluate_population(population):
            return np.array([func(ind) for ind in population])

        def clip_bounds(population):
            return np.clip(population, -5.0, 5.0)

        def mutate(population, target_idx, f):
            candidates = list(range(len(population)))
            candidates.remove(target_idx)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = population[a] + f * (population[b] - population[c])
            return clip_bounds(mutant)

        def adapt_f(population, target_idx, f, prev_best_fitness, prev_best_fitness_count):
            if np.random.rand() < self.p:
                f = np.random.uniform(0, 1)
            elif prev_best_fitness_count > 0:
                improvement_ratio = (prev_best_fitness - prev_best_fitness_count) / prev_best_fitness_count
                f *= improvement_ratio
            return f

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        prev_best_fitness = fitness[best_idx]
        prev_best_fitness_count = 0
        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                f = adapt_f(population, i, self.f, prev_best_fitness, prev_best_fitness_count)
                mutant = mutate(population, i, f)
                trial = crossover(population[i], mutant)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                        prev_best_fitness_count += 1
        return best_solution