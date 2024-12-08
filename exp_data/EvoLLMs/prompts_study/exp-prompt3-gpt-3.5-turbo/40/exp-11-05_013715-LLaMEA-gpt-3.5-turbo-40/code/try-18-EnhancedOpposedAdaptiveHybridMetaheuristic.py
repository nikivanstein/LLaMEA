import numpy as np

class EnhancedOpposedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def opposition(x):
            return -x

        def dynamic_mutate(x, pop, scaling_factor_min=0.1, scaling_factor_max=0.9):
            scaling_factor = scaling_factor_min + np.random.rand() * (scaling_factor_max - scaling_factor_min)
            candidates = pop[np.random.choice(len(pop), 3, replace=False)]
            mutant = x + scaling_factor * (candidates[0] - x) + scaling_factor * (candidates[1] - candidates[2])
            return np.clip(mutant, -5.0, 5.0)

        def adaptive_pso_update(x, pbest, gbest, velocity, inertia_weight_min=0.1, inertia_weight_max=0.9, cognitive_weight_min=1.0, cognitive_weight_max=2.0):
            inertia_weight = inertia_weight_min + np.random.rand() * (inertia_weight_max - inertia_weight_min)
            cognitive_weight = cognitive_weight_min + np.random.rand() * (cognitive_weight_max - cognitive_weight_min)
            social_weight = cognitive_weight_min + np.random.rand() * (cognitive_weight_max - cognitive_weight_min)
            new_velocity = inertia_weight * velocity \
                       + cognitive_weight * np.random.rand(self.dim) * (pbest - x) \
                       + social_weight * np.random.rand(self.dim) * (gbest - x)
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
                mutant = dynamic_mutate(population[i], population)
                trial, new_velocity = adaptive_pso_update(population[i], pbest[i], gbest, initial_velocity)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness

                # Opposition-based learning
                opposition_trial = opposition(trial)
                opposition_trial_fitness = func(opposition_trial)
                if opposition_trial_fitness < gbest_fitness:
                    gbest = opposition_trial
                    gbest_fitness = opposition_trial_fitness

        return gbest