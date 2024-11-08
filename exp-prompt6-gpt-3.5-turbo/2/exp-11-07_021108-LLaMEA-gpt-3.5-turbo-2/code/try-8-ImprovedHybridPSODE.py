import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.pso_params = {'w': 0.5, 'c1': 1.5, 'c2': 1.5}
        self.de_params = {'F': 0.5, 'CR': 0.9}

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = np.full((self.population_size, self.dim), np.inf)
        gbest = np.full(self.dim, np.inf)

        for _ in range(self.budget):
            fitness = np.apply_along_axis(func, 1, particles)
            pbest[fitness < pbest] = particles[fitness < pbest]
            gbest = particles[np.argmin(fitness)] if np.min(fitness) < func(gbest) else gbest

            mutants = particles + self.de_params['F'] * (pbest - particles)[:, None] + self.de_params['F'] * (particles[np.random.choice(self.population_size, size=self.population_size)] - particles[np.random.choice(self.population_size, size=self.population_size)])
            crossover_mask = np.random.rand(self.population_size, self.dim) > self.de_params['CR']
            mutants[crossover_mask] = particles[crossover_mask]
            trials = mutants

            trial_fitness = np.apply_along_axis(func, 1, trials)
            updates = trial_fitness < np.array([func(p) for p in pbest])
            pbest[updates] = trials[updates]

            gbest_update = trial_fitness < func(gbest)
            gbest = trials[np.argmin(trial_fitness)] if np.any(gbest_update) else gbest

            particles = particles

        return gbest