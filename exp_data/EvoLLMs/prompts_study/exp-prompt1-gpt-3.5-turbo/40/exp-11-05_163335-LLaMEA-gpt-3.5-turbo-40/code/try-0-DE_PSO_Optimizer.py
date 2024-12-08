import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.npop = npop
        self.F = F
        self.CR = CR
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def evaluate(pop):
            return np.array([func(ind) for ind in pop])

        def mutate(pbest, gbest, pop):
            mutant_pop = []
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
                if np.random.rand() < self.CR:
                    mutant = np.clip(mutant, -5.0, 5.0)
                else:
                    mutant = pop[i]
                if func(mutant) < func(pop[i]):
                    pop[i] = mutant
                if func(mutant) < func(pbest[i]):
                    pbest[i] = mutant
                if func(mutant) < func(gbest):
                    gbest = mutant
            return pop, pbest, gbest

        def update_velocity(pbest, pop, velocity, gbest):
            for i in range(self.npop):
                velocity[i] = self.w * velocity[i] + self.c1 * np.random.rand() * (pbest[i] - pop[i]) + self.c2 * np.random.rand() * (gbest - pop[i])
                velocity[i] = np.clip(velocity[i], -5.0, 5.0)
            return velocity

        pop = np.random.uniform(-5.0, 5.0, size=(self.npop, self.dim))
        pbest = np.copy(pop)
        gbest = pop[np.argmin(evaluate(pop))]
        velocity = np.zeros_like(pop)
        
        for _ in range(self.budget):
            pop, pbest, gbest = mutate(pbest, gbest, pop)
            velocity = update_velocity(pbest, pop, velocity, gbest)
            pop += velocity
        
        return gbest