import numpy as np

class Hybrid_DE_PSO_Optimizer_Enhanced:
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99):
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

    def __call__(self, func):
        def evaluate(pop):
            return np.array([func(ind) for ind in pop])

        def blend_mutate(pbest, gbest, pop, F, CR, w, c1, c2):
            mutant_pop = []
            for i in range(self.npop):
                idxs = [idx for idx in range(self.npop) if idx != i]
                
                # Differential Evolution Component
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant_de = np.clip(a + F * (b - c), -5.0, 5.0)

                # Particle Swarm Optimization Component
                velocity_i = w * pop[i] + c1 * np.random.rand() * (pbest[i] - pop[i]) + c2 * np.random.rand() * (gbest - pop[i])
                velocity_i = np.clip(velocity_i, -5.0, 5.0)
                
                mutant = np.where(np.random.rand(self.dim) < CR, mutant_de, velocity_i)
                
                if func(mutant) < func(pop[i]):
                    pop[i] = mutant
                if func(mutant) < func(pbest[i]):
                    pbest[i] = mutant
                if func(mutant) < func(gbest):
                    gbest = mutant
            return pop, pbest, gbest

        pop = np.random.uniform(-5.0, 5.0, (self.npop, self.dim))
        pbest = np.copy(pop)
        gbest = pop[np.argmin(evaluate(pop))]
        
        for _ in range(self.budget):
            self.F *= self.F_decay
            self.CR *= self.CR_decay
            self.w *= self.w_decay
            self.c1 *= self.c1_decay
            self.c2 *= self.c2_decay
            
            pop, pbest, gbest = blend_mutate(pbest, gbest, pop, self.F, self.CR, self.w, self.c1, self.c2)
        
        return gbest