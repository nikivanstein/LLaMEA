import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.pso_params = {'w': 0.5, 'c1': 1.5, 'c2': 1.5}
        self.de_params = {'F': 0.5, 'CR': 0.9}

    def __call__(self, func):
        def pso_update(particles, pbest, gbest, func):
            for i in range(self.population_size):
                fitness = func(particles[i])
                if fitness < pbest[i][1]:
                    pbest[i] = (particles[i].copy(), fitness)
                if fitness < gbest[1]:
                    gbest = (particles[i].copy(), fitness)
            return particles, pbest, gbest

        def de_update(particles, pbest, gbest, func):
            for i in range(self.population_size):
                mutant = particles[i] + self.de_params['F'] * (pbest[i][0] - particles[i]) + self.de_params['F'] * (particles[np.random.choice(self.population_size)] - particles[np.random.choice(self.population_size)])
                trial = mutant.copy()
                for j in range(self.dim):
                    if np.random.rand() > self.de_params['CR']:
                        trial[j] = particles[i][j]
                if func(trial) < pbest[i][1]:
                    pbest[i] = (trial.copy(), func(trial))
                if pbest[i][1] < gbest[1]:
                    gbest = pbest[i]
            return particles, pbest, gbest

        particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = [(particle.copy(), np.inf) for particle in particles]
        gbest = (particles[0].copy(), np.inf)

        for _ in range(self.budget):
            particles, pbest, gbest = pso_update(particles, pbest, gbest, func)
            particles, pbest, gbest = de_update(particles, pbest, gbest, func)

        return gbest[0]