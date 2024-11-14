import numpy as np

class HybridDEPSO:
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
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729

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

        def pso_update_velocity(velocity, position, pbest, gbest):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            return self.w * velocity + self.c1 * r1 * (pbest - position) + self.c2 * r2 * (gbest - position)

        def sa_mutation(candidate, best, t):
            self.sigma = max(0.01, min(0.5, self.sigma + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            return candidate + (self.sigma + self.learning_rate) * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        pbest = np.copy(population)
        gbest = np.copy(population[best_idx])
        best_solution = population[best_idx]
        t = 0

        while t < self.budget:
            new_population = np.zeros_like(population)
            new_velocity = np.zeros_like(velocity)
            for i in range(self.pop_size):
                # PSO velocity update
                new_velocity[i] = pso_update_velocity(velocity[i], population[i], pbest[i], gbest)
                
                # PSO position update
                candidate = population[i] + new_velocity[i]
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    pbest[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_solution = candidate
                        best_idx = i
                        gbest = np.copy(candidate)
                else:
                    new_population[i] = sa_mutation(population[i], best_solution, t)
                t += 1

            population = new_population
            velocity = new_velocity

        return best_solution