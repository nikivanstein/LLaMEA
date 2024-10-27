import numpy as np

class Hybrid_DE_PSO_Optimizer:
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

        def de_mutate(population, target_idx, f):
            candidates = list(range(len(population)))
            candidates.remove(target_idx)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = population[a] + f * (population[b] - population[c])
            return clip_bounds(mutant)

        def pso_mutate(population, target_idx, f, personal_best, global_best):
            inertia_weight = self.w
            cognitive_component = self.c1 * np.random.rand(self.dim) * (personal_best - population[target_idx])
            social_component = self.c2 * np.random.rand(self.dim) * (global_best - population[target_idx])
            velocity = inertia_weight * population[target_idx] + cognitive_component + social_component
            return clip_bounds(population[target_idx] + velocity)

        def adapt_f(population, target_idx, f):
            if np.random.rand() < self.p:
                f = np.random.uniform(0, 1)
            return f

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
        personal_best = population[best_idx]
        global_best = best_solution
        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                f = adapt_f(population, i, self.f)
                if np.random.rand() < 0.5:
                    mutant = de_mutate(population, i, f)
                else:
                    mutant = pso_mutate(population, i, f, personal_best, global_best)
                trial = crossover(population[i], mutant)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
            personal_best_idx = np.argmin(fitness)
            if fitness[personal_best_idx] < func(personal_best):
                personal_best = population[personal_best_idx]
            if func(personal_best) < func(global_best):
                global_best = personal_best
        return best_solution