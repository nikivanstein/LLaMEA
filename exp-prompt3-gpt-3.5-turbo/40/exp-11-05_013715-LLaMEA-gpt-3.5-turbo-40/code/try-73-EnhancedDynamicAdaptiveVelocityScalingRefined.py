import numpy as np

class EnhancedDynamicAdaptiveVelocityScalingRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def levy_flight(step_size=0.1):
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, self.dim) ** 2
            v = np.random.normal(0, 1, self.dim)
            step = step_size * v / (u + 1e-10)
            return step

        def crowding_selection(pop, scores):
            sorted_indices = np.argsort(scores)
            crowding_distance = np.zeros(len(pop))
            for i in range(self.dim):
                crowding_distance[sorted_indices[0]] = crowding_distance[sorted_indices[-1]] = np.inf
                for j in range(1, len(pop) - 1):
                    crowding_distance[sorted_indices[j]] += pop[sorted_indices[j + 1], i] - pop[sorted_indices[j - 1], i]
            return pop[sorted_indices[np.argmax(crowding_distance)]]

        def dynamic_adaptive_velocity_scaling(x, pbest, gbest, velocity, w_min=0.4, w_max=0.9, c_min=1.0, c_max=2.0):
            w = w_min + np.random.rand() * (w_max - w_min)
            c1 = c_min + np.random.rand() * (c_max - c_min)
            c2 = c_min + np.random.rand() * (c_max - c_min)
            scaling_factor = np.random.rand() * 2  # Dynamic scaling factor
            new_velocity = scaling_factor * (w * velocity \
                       + c1 * np.random.rand(self.dim) * (pbest - x) \
                       + c2 * np.random.rand(self.dim) * (gbest - x))
            x_new = np.clip(x + new_velocity, -5.0, 5.0)
            return x_new, new_velocity

        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        pbest = population.copy()
        gbest = population[np.argmin(fitness)]
        gbest_fitness = func(gbest)
        initial_velocity = np.zeros(self.dim)

        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                step = levy_flight()
                trial = np.clip(population[i] + step, -5.0, 5.0)
                trial, new_velocity = dynamic_adaptive_velocity_scaling(population[i], pbest[i], gbest, initial_velocity)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness

            gbest = crowding_selection(population, fitness)
            gbest_fitness = func(gbest)

        return gbest