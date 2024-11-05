import numpy as np

class EnhancedTournamentSelectionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def chaotic_mutation(x, pop, f_min=0.4, f_max=0.9):
            f = f_min + 0.5 * (f_max - f_min) * (1 - np.cos(x))
            candidates = pop[np.random.choice(len(pop), 3, replace=False)]
            mutant = x + f * (candidates[0] - x) + f * (candidates[1] - candidates[2]) + np.random.normal(scale=0.1, size=self.dim)
            return np.clip(mutant, -5.0, 5.0)

        def tournament_selection(pop, scores, tournament_size=3):
            selected_indices = []
            for _ in range(len(pop)):
                tournament_indices = np.random.choice(range(len(pop)), tournament_size, replace=False)
                tournament_scores = [scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmin(tournament_scores)]
                selected_indices.append(winner_index)
            return pop[selected_indices]

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
                mutant = chaotic_mutation(population[i], population)
                trial, new_velocity = dynamic_adaptive_velocity_scaling(population[i], pbest[i], gbest, initial_velocity)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness

            gbest = tournament_selection(population, fitness)
            gbest_fitness = func(gbest)

        return gbest