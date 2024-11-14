import numpy as np

class MultiSwarmImprovedHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1
        self.adaptive_param = 0.1  # Adaptive control parameter
        self.learning_rate = 0.05  # Dynamic learning rate
        self.num_swarms = 3  # Number of swarms
        self.swarm_radii = np.full(self.num_swarms, 0.1)  # Initial swarm radii

    def __call__(self, func):
        def de_mutate(population, target_idx):
            candidates = population[np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)]
            self.f = max(0.1, min(0.9, self.f + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            donor_vector = population[target_idx] + (self.f + self.learning_rate) * (candidates[0] - candidates[1])
            for i in range(self.dim):
                if np.random.rand() > self.cr:
                    donor_vector[i] = population[target_idx][i]
            return donor_vector

        def sa_mutation(candidate, best, t):
            self.sigma = max(0.01, min(0.5, self.sigma + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            return candidate + (self.sigma + self.learning_rate) * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        def move_swarm(swarm_center, radius):
            return swarm_center + radius * np.random.normal(0, 1, self.dim)

        swarms = [np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)) for _ in range(self.num_swarms)]
        fitness = [np.array([func(individual) for individual in swarm]) for swarm in swarms]
        best_solutions = [swarm[np.argmin(f)] for swarm, f in zip(swarms, fitness)]
        t = 0

        while t < self.budget:
            new_swarms = []
            for idx, swarm in enumerate(swarms):
                new_swarm = np.zeros_like(swarm)
                for i in range(self.pop_size):
                    candidate = de_mutate(swarm, i)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[idx][i]:
                        new_swarm[i] = candidate
                        fitness[idx][i] = candidate_fitness
                        if candidate_fitness < func(best_solutions[idx]):
                            best_solutions[idx] = candidate
                    else:
                        new_swarm[i] = sa_mutation(swarm[i], best_solutions[idx], t)
                    t += 1
                new_swarms.append(new_swarm)

            for idx in range(self.num_swarms):
                swarms[idx] = move_swarm(np.mean(new_swarms[idx], axis=0), self.swarm_radii[idx])

        return best_solutions[np.argmin([func(sol) for sol in best_solutions])]