def dynamic_local_search_rate(fitness_history, local_search_rate, max_rate=0.5, min_rate=0.1, rate_change=0.05):
    if len(fitness_history) > 1 and fitness_history[-1] < fitness_history[-2]:
        local_search_rate = max(min_rate, local_search_rate - rate_change)
    else:
        local_search_rate = min(max_rate, local_search_rate + rate_change)
    return local_search_rate

class EnhancedDynamicHarmonySearchFasterConvergence(EnhancedDynamicHarmonySearch):
    def __call__(self, func):
        # Existing code remains the same
        
        for _ in range(self.budget):
            # Existing code remains the same
            
            self.local_search_rate = dynamic_local_search_rate(fitness_history, self.local_search_rate)  # New line
        
        return best_solution