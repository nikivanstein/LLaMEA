class Enhanced_EA_ADES(Improved_EA_ADES):
    def __call__(self, func):
        def mutate(x, pop, strategy):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            scaling_factor = np.random.uniform(0.5, 1.0) if np.random.rand() < self.mut_prob else self.f
            return x + scaling_factor * (b - c)