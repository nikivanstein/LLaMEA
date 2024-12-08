import numpy as np

class APSO_DIVC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w_max = 0.9  # maximum inertia
        self.w_min = 0.4  # minimum inertia
        self.v_max = (self.upper_bound - self.lower_bound) / 2.0  # max velocity

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        personal_best = np.copy(population)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best)
        self.evaluations = self.pop_size

        best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[best_idx]
        global_best_fitness = personal_best_fitness[best_idx]

        while self.evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))

            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (personal_best[i] - population[i])
                social = self.c2 * r2 * (global_best - population[i])

                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

                current_fitness = func(population[i])
                self.evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = current_fitness

        return global_best, global_best_fitness