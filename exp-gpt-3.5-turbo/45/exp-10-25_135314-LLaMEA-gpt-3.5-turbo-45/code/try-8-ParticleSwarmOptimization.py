import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget, dim, inertia=0.5, cognitive_weight=1.5, social_weight=2.0):
        self.budget = budget
        self.dim = dim
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        self.velocities = np.zeros((self.budget, self.dim))
        self.personal_bests = self.population.copy()
        self.global_best = self.personal_bests[np.argmin([func(individual) for individual in self.personal_bests])]

    def update_particle(self, particle, personal_best):
        cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (personal_best - particle)
        social_component = self.social_weight * np.random.rand(self.dim) * (self.global_best - particle)
        return particle + self.inertia * self.velocities[i] + cognitive_component + social_component

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                self.velocities[i] = self.update_particle(self.population[i], self.personal_bests[i])
                self.population[i] = np.clip(self.population[i] + self.velocities[i], -5.0, 5.0)
                if func(self.population[i]) < func(self.personal_bests[i]):
                    self.personal_bests[i] = self.population[i]
                    if func(self.population[i]) < func(self.global_best):
                        self.global_best = self.population[i]

        return self.global_best