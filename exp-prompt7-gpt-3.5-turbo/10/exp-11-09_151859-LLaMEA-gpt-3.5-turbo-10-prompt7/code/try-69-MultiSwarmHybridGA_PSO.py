import numpy as np

class MultiSwarmHybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, num_swarms=5, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.num_swarms = num_swarms
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        global_bests = [swarm[np.argmin([func(ind) for ind in swarm])] for swarm in swarms]
        velocities = [np.zeros((self.population_size, self.dim)) for _ in range(self.num_swarms)]

        mutation_rate_adjustment = 0.1
        for _ in range(self.budget):
            for i in range(self.num_swarms):
                best_individual = swarms[i][np.argmin([func(ind) for ind in swarms[i]])]
                velocities[i] = self.inertia_weight * velocities[i] + self.cognitive_weight * np.random.rand() * (best_individual - swarms[i]) + self.social_weight * np.random.rand() * (global_bests[i] - swarms[i])
                swarms[i] += velocities[i]
                mutation_rate_adjustment = 0.1 * np.exp(-_ / self.budget)
                mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
                swarms[i] = swarms[i] + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask

                if func(swarms[i][np.argmin([func(ind) for ind in swarms[i]])]) < func(global_bests[i]):
                    global_bests[i] = swarms[i][np.argmin([func(ind) for ind in swarms[i]])]

        return global_bests[np.argmin([func(ind) for ind in global_bests])]