import nympy as np

def softmax(z):
	return np.exp(z) / sum(np.exp(z))
