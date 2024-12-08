import numpy as np

class ImprovedOpposedChaoticHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def chaotic_mutation(x, pop, f_min=0.4, f_max=0.9):
            f = f_min + 0.5 * (f_max - f_min) * (1 - np.cos(x))
            candidates = pop[np.random.choice(len(pop), 3, replace=False)]
            mutant = x + f * (candidates[0] - x) + f * (candidates[1] - candidates[2])
            return np.clip(mutant, -5.0, 5.0)

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
                mutant = chaotic_mutation(population[i], population)
                trial, new_velocity = adaptive_pso_update(population[i], pbest[i], gbest, initial_velocity)
                trial_fitness = func(trial)

                # Introducing dynamic opposition-based learning
                if np.random.rand() < 0.5:  
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