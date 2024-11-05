import numpy as np

class EnhancedOpposedChaoticDEMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def differential_evolution_mutation(population, i, mut=0.8, crossp=0.7):
            candidates = [ind for ind in range(len(population)) if ind != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), -5.0, 5.0)
            return mutant

        def adaptive_pso_update(x, pbest, gbest, velocity, w_min=0.4, w_max=0.9, c_min=1.0, c_max=2.0):
            w = w_min + np.random.rand() * (w_max - w_min)
            c1 = c_min + np.random.rand() * (c_max - c_min)
            c2 = c_min + np.random.rand() * (c_max - c_min)
            new_velocity = w * velocity \
                           + c1 * np.random.rand(self.dim) * (pbest - x) \
                           + c2 * np.random.rand(self.dim) * (gbest - x)
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
                mutant = differential_evolution_mutation(population, i)
                trial, new_velocity = adaptive_pso_update(population[i], pbest[i], gbest, initial_velocity)
                trial_fitness = func(trial)
                opposite_trial = 2 * gbest - trial
                opposite_trial_fitness = func(opposite_trial)
                if opposite_trial_fitness < fitness[i]:
                    population[i] = opposite_trial
                    fitness[i] = opposite_trial_fitness
                    pbest[i] = opposite_trial
                    if opposite_trial_fitness < gbest_fitness:
                        gbest = opposite_trial
                        gbest_fitness = opposite_trial_fitness
                else:
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        pbest[i] = trial
                        if trial_fitness < gbest_fitness:
                            gbest = trial
                            gbest_fitness = trial_fitness

        return gbest