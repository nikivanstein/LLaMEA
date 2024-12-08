class ImprovedHybridFireflyDE(LocalSearchHybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1, search_radius=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size, adapt_rate, search_radius)

    def __call__(self, func):
        # Same as before with the addition of local search strategy
        
        while budget_used < self.budget:
            for i in range(self.pop_size):
                # Existing code for firefly algorithm
                # Local search strategy
                local_search = clipToBounds(pop[i] + np.random.uniform(-self.search_radius, self.search_radius, self.dim))
                
                if func(local_search) < func(pop[i]):
                    pop[i] = local_search
                    budget_used += 1

                if budget_used >= self.budget:
                    break

        return pop[np.argmin([func(ind) for ind in pop])]