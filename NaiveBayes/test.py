from NaiveBayes import MultinomialNB,GaussianNB
import numpy as np


if __name__ == "__main__":
    X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
                              ])
    X = X.T
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

    nb = MultinomialNB(alpha=1.0,fit_prior=True)
    nb.fit(X,y)
    print nb.alpha
    print nb.class_prior
    print nb.classes
    print nb.conditional_prob
    print nb.predict(np.array([2,4]))

    nb1 = GaussianNB(alpha=0.0)
    print nb1.fit(X,y).predict(X)

