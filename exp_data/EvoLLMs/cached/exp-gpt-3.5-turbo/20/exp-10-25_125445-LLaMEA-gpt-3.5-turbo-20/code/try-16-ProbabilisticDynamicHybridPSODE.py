import numpy as np

class ProbabilisticDynamicHybridPSODE(DynamicHybridPSODE):
    def __init__(self, budget, dim, population_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9, min_w=0.4, max_w=0.9, min_c=0.5, max_c=2.0, min_f=0.3, max_f=0.8, min_cr=0.5, max_cr=1.0, prob=0.2):
        super().__init__(budget, dim, population_size, w, c1, c2, f, cr, min_w, max_w, min_c, max_c, min_f, max_f, min_cr, max_cr)
        self.prob = prob

    def probabilistic_adapt_parameters(self, func, gbest_val):
        improvement_threshold = 0.1
        if np.random.rand() < self.prob:
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
            self.probabilistic_adapt_parameters(func, gbest['value'])
        return gbest['value']