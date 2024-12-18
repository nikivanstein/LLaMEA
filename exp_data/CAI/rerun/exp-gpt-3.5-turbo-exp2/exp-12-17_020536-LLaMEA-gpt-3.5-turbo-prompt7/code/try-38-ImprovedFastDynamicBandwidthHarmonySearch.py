from joblib import Parallel, delayed

class ImprovedFastDynamicBandwidthHarmonySearch(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        for _ in range(self.budget):
            new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for _ in range(2):
                pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))  # Dynamic pitch rate adjustment
                def evaluate_candidate(i):
                    new_harmony_local = np.copy(self.harmony_memory[-1])
                    new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth), min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
                    return i, func(new_harmony_local)
                candidate_results = Parallel(n_jobs=-1)(delayed(evaluate_candidate)(i) for i in range(self.dim))
                for i, result in candidate_results:
                    if result < func(self.harmony_memory[-1]):
                        self.harmony_memory[-1][i] = new_harmony_local[i]
            bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]