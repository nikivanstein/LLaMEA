import numpy as np

class EnhancedAdaptivePopulationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.base_mutation_rate = 0.1
        self.mutation_rate = self.base_mutation_rate
        self.min_particles = 10
        self.max_particles = 100

    def __call__(self, func):
        num_particles = self.min_particles
        swarm = np.random.uniform(-5.0, 5.0, (num_particles, self.dim))
        velocity = np.zeros((num_particles, self.dim))
        pbest = swarm.copy()
        pbest_fitness = np.apply_along_axis(func, 1, pbest)
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)

        w = self.w_max

        for t in range(1, self.budget + 1):
            r1 = np.random.random((num_particles, self.dim))
            r2 = np.random.random((num_particles, self.dim))

            velocity = w * velocity + self.c1 * r1 * (pbest - swarm) + self.c2 * r2 * (gbest - swarm)
            swarm += velocity

            fitness = np.apply_along_axis(func, 1, swarm)
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = swarm[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            if np.min(fitness) < gbest_fitness:
                gbest = swarm[np.argmin(fitness)]
                gbest_fitness = np.min(fitness)

            w = self.w_min + (t / self.budget) * (self.w_max - self.w_min)

            # Adaptive Population Size
            mean_fitness = np.mean(fitness)
            if t % (self.budget // 5) == 0:  # Adjust population size every 20% of the budget
                if mean_fitness < gbest_fitness * 1.05 and num_particles < self.max_particles:
                    num_particles += 10
                    swarm = np.vstack((swarm, np.random.uniform(-5.0, 5.0, (10, self.dim)))
                    velocity = np.vstack((velocity, np.zeros((10, self.dim)))
                    pbest = np.vstack((pbest, np.random.uniform(-5.0, 5.0, (10, self.dim)))
                    pbest_fitness = np.hstack((pbest_fitness, np.apply_along_axis(func, 1, pbest[-10:])))
                elif mean_fitness > gbest_fitness * 1.1 and num_particles > self.min_particles:
                    num_particles -= 10
                    swarm = swarm[:num_particles]
                    velocity = velocity[:num_particles]
                    pbest = pbest[:num_particles]
                    pbest_fitness = pbest_fitness[:num_particles]

        return gbest_fitness