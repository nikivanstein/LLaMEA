import numpy as np

class Enhanced_Dynamic_DE_PSO_Optimizer(Dynamic_DE_PSO_Optimizer):
    def __init__(self, budget, dim, npop=30, F=0.5, CR=0.9, w=0.7, c1=1.5, c2=1.5, F_decay=0.95, CR_decay=0.95, w_decay=0.99, c1_decay=0.99, c2_decay=0.99):
        super().__init__(budget, dim, npop, F, CR, w, c1, c2, F_decay, CR_decay, w_decay, c1_decay, c2_decay)

    def mutate_hybrid(pbest, gbest, pop, F, CR):
        mutant_pop = []
        for i in range(self.npop):
            idxs = [idx for idx in range(self.npop) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            pbest_mutant = np.clip(pop[i] + F * (pbest[i] - pop[i]) + F * (gbest - pop[i]), -5.0, 5.0)
            if np.random.rand() < CR:
                mutant = np.clip(pbest_mutant, -5.0, 5.0)
            else:
                mutant = pop[i]
            if func(mutant) < func(pop[i]):
                pop[i] = mutant
            if func(mutant) < func(pbest[i]):
                pbest[i] = mutant
            if func(mutant) < func(gbest):
                gbest = mutant
        return pop, pbest, gbest

    for _ in range(self.budget):
        self.F *= self.F_decay
        self.CR *= self.CR_decay
        self.w *= self.w_decay
        self.c1 *= self.c1_decay
        self.c2 *= self.c2_decay
        
        pop, pbest, gbest = mutate_hybrid(pbest, gbest, pop, self.F, self.CR)
        velocity = update_velocity(pbest, pop, velocity, gbest, self.w, self.c1, self.c2)
        pop += velocity
    
    return gbest