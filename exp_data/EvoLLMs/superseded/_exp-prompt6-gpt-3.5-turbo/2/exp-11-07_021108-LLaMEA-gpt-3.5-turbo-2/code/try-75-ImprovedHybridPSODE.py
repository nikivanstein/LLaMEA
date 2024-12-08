import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.pso_params = {'w': 0.5, 'c1': 1.5, 'c2': 1.5}
        self.de_params = {'F': 0.5, 'CR': 0.9}

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        pbest_particles = particles.copy()
        pbest_values = np.full(self.population_size, np.inf)
        gbest_particle = particles[0].copy()
        gbest_value = np.inf

        for _ in range(self.budget):
            fitness_values = np.apply_along_axis(func, 1, particles)

            pbest_update_mask_pso = fitness_values < pbest_values
            pbest_values[pbest_update_mask_pso] = fitness_values[pbest_update_mask_pso]
            pbest_particles[pbest_update_mask_pso] = particles[pbest_update_mask_pso]

            gbest_index = np.argmin(fitness_values)
            gbest_value = np.where(fitness_values[gbest_index] < gbest_value, fitness_values[gbest_index], gbest_value)
            gbest_particle = np.where(fitness_values[gbest_index] < gbest_value, particles[gbest_index].copy(), gbest_particle)

            mutants = particles + self.de_params['F'] * (pbest_particles - particles) + self.de_params['F'] * (
                        particles[np.random.choice(self.population_size, size=self.population_size, replace=True)] - particles[
                    np.random.choice(self.population_size, size=self.population_size, replace=True)])
            trials = mutants.copy()

            cr_mask = np.random.rand(self.population_size, self.dim) > self.de_params['CR']
            trials = np.where(cr_mask[:, np.newaxis], particles, trials)

            trial_fitness_values = np.apply_along_axis(func, 1, trials)
            pbest_update_mask_de = trial_fitness_values < pbest_values
            pbest_values[pbest_update_mask_de] = trial_fitness_values[pbest_update_mask_de]
            pbest_particles[pbest_update_mask_de] = trials[pbest_update_mask_de]

            gbest_index_de = np.argmin(trial_fitness_values)
            gbest_value = np.where(trial_fitness_values[gbest_index_de] < gbest_value, trial_fitness_values[gbest_index_de], gbest_value)
            gbest_particle = np.where(trial_fitness_values[gbest_index_de] < gbest_value, trials[gbest_index_de].copy(), gbest_particle)

            particles = trials

        return gbest_particle