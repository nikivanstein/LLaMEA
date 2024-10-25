# import numpy as np

class PSOSimulatedAnnealingDE(PSOSimulatedAnnealing):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.cr = 0.5
        self.f = 0.5

    def __call__(self, func):
        def update_position_de(particle, pbest):
            r1, r2, r3 = np.random.choice(range(len(particles)), 3, replace=False)
            mutant = particles[r1]['position'] + self.f * (particles[r2]['position'] - particles[r3]['position'])
            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, particle['position'])
            velocity = particle['velocity']
            new_velocity = self.inertia_weight * velocity + self.cognitive_weight * np.random.rand() * (pbest['position'] - particle['position']) + self.social_weight * np.random.rand() * (gbest['position'] - particle['position'])
            new_position = particle['position'] + new_velocity
            return new_position

        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim)} for _ in range(self.swarm_size)]
        gbest = min(particles, key=lambda p: func(p['position']))

        for _ in range(self.budget):
            for particle in particles:
                particle['position'] = update_position_de(particle, particle)
                particle['position'] = simulated_annealing(particle['position'], func)

        gbest = min(particles, key=lambda p: func(p['position']))
        
        return gbest['position']