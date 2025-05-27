import numpy as np




class FitnessEvaluator:
    """Class to compute fitness functions for balancing FPS and MCC."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=0.1, epsilon=0.01, lambda_=5.0, target_fps=120.0):
        """
        Initializes the fitness evaluator with tunable parameters.
        
        Args:
            alpha (float): Weight for FPS in the weighted sum formula.
            beta (float): Weight for MCC in the weighted sum formula.
            gamma (float): Exponential scaling factor for MCC penalty.
            delta (float): Bias term to avoid MCC reaching zero.
            epsilon (float): Small value for logarithmic stability.
            lambda_ (float): Sigmoid steepness in inverse MCC penalty.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.target_fps = target_fps

    @staticmethod
    def rebound_metrics(fps, mcc, target_fps=120.0):
        fps_ratio = min(fps / target_fps, 1.0)  # Cap at 1.0 to avoid rewarding excessively high FPS
        #mcc_norm = (mcc + 1) / 2  # Normalizing MCC from [-1, 1] to [0, 1]
        mcc_norm = mcc
        return fps_ratio, mcc_norm


    def weighted_sum_exponential(self, fps, mcc):
        """Computes fitness using weighted sum with exponential MCC penalty."""
        fps, mcc = self.rebound_metrics(fps, mcc, self.target_fps)
        return self.alpha * fps + self.beta * mcc * np.exp(self.gamma * mcc)

    def multiplicative_penalty(self, fps, mcc):
        """Computes fitness using FPS multiplied by MCC with a bias."""
        fps, mcc = self.rebound_metrics(fps, mcc, self.target_fps)
        return fps * max(0, mcc + self.delta)

    def logarithmic_penalty(self, fps, mcc):
        """Computes fitness using logarithmic MCC scaling."""
        fps, mcc = self.rebound_metrics(fps, mcc, self.target_fps)
        return fps * np.log(1 + max(0, mcc + self.epsilon))

    def inverse_mcc_penalty(self, fps, mcc):
        """Computes fitness by scaling FPS using a sigmoid function of MCC."""
        fps, mcc = self.rebound_metrics(fps, mcc, self.target_fps)
        return fps / (1 + np.exp(-self.lambda_ * mcc))

    def stepwise_penalty(self, fps, mcc):
        """Computes fitness with a stepwise penalty for negative MCC."""
        fps, mcc = self.rebound_metrics(fps, mcc, self.target_fps)
        return fps * (1 + mcc) if mcc >= 0 else fps * 0.1



if __name__ == "__main__":   

    # Example usage:
    evaluator = FitnessEvaluator()
    fps = 536.47
    mcc = -0.00926

    print("Weighted Sum Exponential:", evaluator.weighted_sum_exponential(fps, mcc))
    print("Multiplicative Penalty:", evaluator.multiplicative_penalty(fps, mcc))
    print("Logarithmic Penalty:", evaluator.logarithmic_penalty(fps, mcc))
    print("Inverse MCC Penalty:", evaluator.inverse_mcc_penalty(fps, mcc))
    print("Stepwise Penalty:", evaluator.stepwise_penalty(fps, mcc))