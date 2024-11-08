import numpy as np

class ImprovedHybridPSODEEfficient:
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
            pbest_masked = particles[update_mask_pso]
            pbest_values[update_mask_pso] = fitness_values[update_mask_pso]
            pbest_particles[update_mask_pso] = pbest_masked

            gbest_index = np.argmin(fitness_values)
            if fitness_values[gbest_index] < gbest_value:
                gbest_particle = particles[gbest_index].copy()
                gbest_value = fitness_values[gbest_index]

            random_choices = np.random.choice(self.population_size, size=self.population_size, replace=True)
            mutants = particles + self.de_params['F'] * (pbest_particles - particles) + self.de_params['F'] * (
                        particles[random_choices] - particles[random_choices])
            trials = mutants.copy()

            cr_mask = np.random.rand(self.population_size, self.dim) > self.de_params['CR']
            cr_where = np.where(cr_mask, particles, trials)
            for i in range(self.population_size):
                trials[i] = cr_where[i]

            trial_fitness_values = np.apply_along_axis(func, 1, trials)
            update_mask_de = trial_fitness_values < pbest_values
            update_masked = trials[update_mask_de]
            pbest_values[update_mask_de] = trial_fitness_values[update_mask_de]
            pbest_particles[update_mask_de] = update_masked

            gbest_index_de = np.argmin(trial_fitness_values)
            if trial_fitness_values[gbest_index_de] < gbest_value:
                gbest_particle = trials[gbest_index_de].copy()
                gbest_value = trial_fitness_values[gbest_index_de]

            particles = trials

        return gbest_particle