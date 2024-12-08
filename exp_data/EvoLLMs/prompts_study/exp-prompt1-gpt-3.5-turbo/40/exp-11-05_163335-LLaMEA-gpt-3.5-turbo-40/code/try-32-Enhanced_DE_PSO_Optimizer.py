import numpy as np

class Enhanced_DE_PSO_Optimizer:
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99, alpha=0.2, beta=0.2):
        self.budget = budget
        self.dim = dim
        self.npop = npop
        self.F = F
        self.CR = CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.F_decay = F_decay
        self.CR_decay = CR_decay
        self.w_decay = w_decay
        self.c1_decay = c1_decay
        self.c2_decay = c2_decay
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        def evaluate(pop):
            return np.array([func(ind) for ind in pop])

        def mutate(pbest, gbest, pop, F, CR):
            mutant_pop = []
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                if np.random.rand() < CR:
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

        def update_velocity(pbest, pop, velocity, gbest, w, c1, c2):
            for i in range(self.npop):
                velocity[i] = w * velocity[i] + c1 * np.random.rand() * (pbest[i] - pop[i]) + c2 * np.random.rand() * (gbest - pop[i])
                velocity[i] = np.clip(velocity[i], -5.0, 5.0)
            return velocity

        pop = np.random.uniform(-5.0, 5.0, size=(self.npop, self.dim))
        pbest = np.copy(pop)
        gbest = pop[np.argmin(evaluate(pop))]
        velocity = np.zeros_like(pop)
        
        for _ in range(self.budget):
            self.F *= self.F_decay
            self.CR *= self.CR_decay
            self.w *= self.w_decay
            self.c1 *= self.c1_decay
            self.c2 *= self.c2_decay
            
            pop, pbest, gbest = mutate(pbest, gbest, pop, self.F, self.CR)
            velocity = update_velocity(pbest, pop, velocity, gbest, self.w, self.c1, self.c2)
            pop += velocity
            self.F += self.alpha * np.random.randn()
            self.CR += self.beta * np.random.randn()
            self.w += self.alpha * np.random.randn()
        
        return gbest