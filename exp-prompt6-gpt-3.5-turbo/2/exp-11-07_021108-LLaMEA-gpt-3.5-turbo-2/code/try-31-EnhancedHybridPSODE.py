import numpy as np

class EnhancedHybridPSODE:
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

            update_mask_pso = fitness_values < pbest_values
            pbest_values[update_mask_pso] = fitness_values[update_mask_pso]
            pbest_particles[update_mask_pso] = particles[update_mask_pso]

            gbest_index = np.argmin(fitness_values)
            gbest_mask = fitness_values < gbest_value  # Replaced nested loop with vectorized operation
            gbest_particle = np.where(gbest_mask, particles[gbest_index].copy(), gbest_particle)
            gbest_value = np.where(gbest_mask, fitness_values[gbest_index], gbest_value)

            mutants = particles + self.de_params['F'] * (pbest_particles - particles) + self.de_params['F'] * (
                        particles[np.random.choice(self.population_size, size=self.population_size, replace=True)] - particles[
                    np.random.choice(self.population_size, size=self.population_size, replace=True)])
            trials = mutants.copy()

            cr_mask = np.random.rand(self.population_size, self.dim) > self.de_params['CR']
            cr_mask_reshaped = np.expand_dims(cr_mask, axis=2)
            cr_masked_trials = np.where(cr_mask_reshaped, particles, trials)
            trial_fitness_values = np.apply_along_axis(func, 1, cr_masked_trials)

            update_mask_de = trial_fitness_values < pbest_values
            pbest_values[update_mask_de] = trial_fitness_values[update_mask_de]
            pbest_particles[update_mask_de] = trials[update_mask_de]

            gbest_index_de = np.argmin(trial_fitness_values)
            gbest_mask_de = trial_fitness_values < gbest_value  # Replaced nested loop with vectorized operation
            gbest_particle = np.where(gbest_mask_de, trials[gbest_index_de].copy(), gbest_particle)
            gbest_value = np.where(gbest_mask_de, trial_fitness_values[gbest_index_de], gbest_value)

            particles = trials

        return gbest_particle