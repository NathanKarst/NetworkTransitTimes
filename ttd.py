import numpy as np

class TransitTimeDistribution:
    def __init__(self,paths,delays,probabilities):
        self.paths = paths
        self.delays = np.array(delays)
        self.probs = np.array(probabilities)
        
        self.compute_expected_delay()
        
    def compute_expected_delay(self):
        self.expected_delay = 0
        for i in range(len(self.delays)):
            self.expected_delay += self.probs[i]*self.delays[i]
            
    def plot(self,kind='pdf',normalizer=1,xlabel='Transit Time',alpha=0.25):
        taus = self.delays/normalizer

        if kind == 'pdf':  
            plt.loglog(taus,self.probs,'ko')
            plt.xlabel(xlabel)
            plt.ylabel('Probability []')
        elif kind == 'cdf':
            idx = np.argsort(taus)

            taus = taus[idx]
            probs = self.probs[idx]
            probs = np.cumsum(probs)

            plt.plot(taus,probs,'k-',alpha=alpha)
            plt.xlabel(xlabel)
            plt.ylabel('Cumul. Prob. []')            