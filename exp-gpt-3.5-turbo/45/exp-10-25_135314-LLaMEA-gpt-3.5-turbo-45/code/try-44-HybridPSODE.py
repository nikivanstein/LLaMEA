import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, w=0.7, c1=1.5, c2=2.0, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.F = F
        self.CR = CR

    def update_velocity(self, velocity, position, pbest, gbest):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        return self.w * velocity + self.c1 * r1 * (pbest - position) + self.c2 * r2 * (gbest - position)

    def ensure_bounds(self, position):
        position[position > 5.0] = 5.0
        position[position < -5.0] = -5.0
        return position

    def mutate(self, population, pbest, gbest):
        mutant_population = []
        for i, p in enumerate(population):
            idxs = np.random.choice(range(len(population)), 3, replace=False)
            target_vector = p + self.F * (population[idxs[0]] - p) + self.F * (population[idxs[1]] - population[idxs[2]])
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_vector = np.where(cross_points, target_vector, p)
            trial_vector = self.ensure_bounds(trial_vector)
            if func(trial_vector) < func(pbest[i]):
                pbest[i] = trial_vector
            if func(trial_vector) < func(gbest):
                gbest = trial_vector
            mutant_population.append(trial_vector)
        return np.array(mutant_population), pbest, gbest

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        pbest = population.copy()
        gbest = pbest[np.argmin([func(individual) for individual in pbest])]

        for _ in range(self.budget):
            velocity = self.update_velocity(velocity, population, pbest, gbest)
            population += velocity
            population = self.ensure_bounds(population)

            population, pbest, gbest = self.mutate(population, pbest, gbest)

        return gbest