import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iters = budget // self.population_size

    def __call__(self, func):
        def mutate(x, pbest, gbest, c1=1.5, c2=1.5, f=0.5):
            v = x + c1 * np.random.random() * (pbest - x) + c2 * np.random.random() * (gbest - x)
            return x + f * (v - x)

        def differential_evolution(population, f=0.5, cr=0.9):
            new_population = []
            for i, target in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant_vector = a + f * (b - c)
                trial_vector = np.where(np.random.random(self.dim) < cr, mutant_vector, target)
                new_population.append(trial_vector if func(trial_vector) < func(target) else target)
            return new_population

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        pbest = population.copy()
        gbest = population[np.argmin([func(ind) for ind in population])]

        for _ in range(self.max_iters):
            for idx, particle in enumerate(population):
                new_particle = mutate(particle, pbest[idx], gbest)
                population[idx] = new_particle if func(new_particle) < func(particle) else particle

            pbest = np.array([p if func(p) < func(old_p) else old_p for p, old_p in zip(population, pbest)])
            gbest = population[np.argmin([func(ind) for ind in population])]

            population = differential_evolution(population)

        return gbest