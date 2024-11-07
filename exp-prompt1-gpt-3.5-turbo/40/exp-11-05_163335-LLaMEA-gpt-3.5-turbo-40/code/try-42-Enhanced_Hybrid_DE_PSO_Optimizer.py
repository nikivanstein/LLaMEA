import numpy as np

class Enhanced_Hybrid_DE_PSO_Optimizer:
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99, F_min=0.1, F_max=0.9, CR_min=0.1, CR_max=0.9, w_min=0.5, w_max=1.0, c1_min=0.5, c1_max=2.0, c2_min=0.5, c2_max=2.0):
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
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.w_min = w_min
        self.w_max = w_max
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2_min = c2_min
        self.c2_max = c2_max

    def __call__(self, func):
        def evaluate(pop):
            return np.array([func(ind) for ind in pop])

        def adapt_parameters():
            self.F = np.clip(np.random.normal(self.F, 0.1), self.F_min, self.F_max)
            self.CR = np.clip(np.random.normal(self.CR, 0.1), self.CR_min, self.CR_max)
            self.w = np.clip(np.random.normal(self.w, 0.1), self.w_min, self.w_max)
            self.c1 = np.clip(np.random.normal(self.c1, 0.1), self.c1_min, self.c1_max)
            self.c2 = np.clip(np.random.normal(self.c2, 0.1), self.c2_min, self.c2_max)

        def mutate(pbest, gbest, pop, F, CR):
            mutant_pop = []
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant_rand = np.clip(a + F * (b - c), -5.0, 5.0)
                
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant_best = np.clip(pop[i] + F * (gbest - pop[i]) + F * (a - b), -5.0, 5.0)
                
                mutant = np.where(np.random.rand(self.dim) < CR, mutant_rand, mutant_best)
                
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
            adapt_parameters()
            
            pop, pbest, gbest = mutate(pbest, gbest, pop, self.F, self.CR)
            velocity = update_velocity(pbest, pop, velocity, gbest, self.w, self.c1, self.c2)
            pop += velocity
        
        return gbest
