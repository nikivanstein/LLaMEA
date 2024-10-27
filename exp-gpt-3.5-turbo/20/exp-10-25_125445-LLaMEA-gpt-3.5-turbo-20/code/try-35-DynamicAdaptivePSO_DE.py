import numpy as np

class DynamicAdaptivePSO_DE:
    def __init__(self, budget, dim, population_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9, min_w=0.4, max_w=0.9, min_c=0.5, max_c=2.0, min_f=0.3, max_f=0.8, min_cr=0.5, max_cr=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr
        self.min_w = min_w
        self.max_w = max_w
        self.min_c = min_c
        self.max_c = max_c
        self.min_f = min_f
        self.max_f = max_f
        self.min_cr = min_cr
        self.max_cr = max_cr

    def pso_update(self, swarm, func):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = self.w * swarm[i]['velocity'] + self.c1 * r1 * (swarm[i]['pbest_pos'] - swarm[i]['position']) + self.c2 * r2 * (swarm[i]['gbest_pos'] - swarm[i]['position'])
            position = swarm[i]['position'] + velocity
            if func(position) < func(swarm[i]['pbest_pos']):
                swarm[i]['pbest_pos'] = position.copy()
                swarm[i]['pbest_val'] = func(position)
            swarm[i]['position'] = position.copy()
        return swarm

    def de_update(self, population, func):
        for i in range(self.population_size):
            x, a, b, c = population[i]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position']
            mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) <= self.cr, mutant, x)
            if func(trial) < func(x):
                population[i]['position'] = trial.copy()
        return population

    def adapt_parameters(self, func, gbest_val):
        improvement_threshold = 0.1
        if np.random.rand() < 0.2:
            if gbest_val < improvement_threshold:
                self.w = np.clip(self.w * 1.1, self.min_w, self.max_w)
                self.c1 = np.clip(self.c1 * 1.1, self.min_c, self.max_c)
                self.c2 = np.clip(self.c2 * 1.1, self.min_c, self.max_c)
                self.f = np.clip(self.f * 1.1, self.min_f, self.max_f)
                self.cr = np.clip(self.cr * 1.1, self.min_cr, self.max_cr)

    def __call__(self, func):
        swarm = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim), 'pbest_pos': np.zeros(self.dim), 'pbest_val': np.inf, 'gbest_pos': np.zeros(self.dim), 'gbest_val': np.inf} for _ in range(self.population_size)]
        gbest = {'position': np.zeros(self.dim), 'value': np.inf}
        for _ in range(self.budget):
            swarm = self.pso_update(swarm, func)
            for particle in swarm:
                if func(particle['position']) < gbest['value']:
                    gbest['position'] = particle['position'].copy()
                    gbest['value'] = func(particle['position'])
            swarm = self.de_update(swarm, func)
            self.adapt_parameters(func, gbest['value'])
        return gbest['value']