import numpy as np

class Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.max_iter = 1000
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.5
        self.temp = 10.0
        self.alpha = 0.95
        self.min_pop_size = 5
        self.max_pop_size = 20
        self.population_history = []

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def accept_move(cost_diff, temp):
            if cost_diff < 0:
                return True
            return np.random.uniform(0, 1) < np.exp(-cost_diff / temp)

        particles = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best = particles.copy()
        global_best = particles[np.argmin([objective_function(p) for p in particles])

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + self.c1 * r1 * (personal_best[i] - particles[i]) + self.c2 * r2 * (global_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)

                for _ in range(5):
                    new_particle = particles[i] + np.random.normal(0, self.temp, size=self.dim)
                    cost_diff = objective_function(new_particle) - objective_function(particles[i])
                    if accept_move(cost_diff, self.temp):
                        particles[i] = new_particle

                if objective_function(particles[i]) < objective_function(personal_best[i]):
                    personal_best[i] = particles[i].copy()
                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()

            self.temp *= self.alpha

            self.population_history.append(self.pop_size)
            if len(self.population_history) > 10:
                if np.std(self.population_history[-10:]) < 2.0 and self.pop_size > self.min_pop_size:
                    self.pop_size -= 1
                elif np.std(self.population_history[-10:]) > 5.0 and self.pop_size < self.max_pop_size:
                    self.pop_size += 1

        return global_best