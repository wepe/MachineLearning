"""
Created on 2015/09/06

@author: wepon (http://2hwp.com)

API Reference: http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
"""
import numpy as np

class MultinomialNB(object):
        """
	Naive Bayes classifier for multinomial models
        The multinomial Naive Bayes classifier is suitable for classification with
        discrete features
	
	Parameters
        ----------
        alpha : float, optional (default=1.0)
                Setting alpha = 0 for no smoothing
		Setting 0 < alpha < 1 is called Lidstone smoothing
		Setting alpha = 1 is called Laplace smoothing 
        fit_prior : boolean
                Whether to learn class prior probabilities or not.
                If false, a uniform prior will be used.
        class_prior : array-like, size (n_classes,)
                Prior probabilities of the classes. If specified the priors are not
                adjusted according to the data.
		
	Attributes
        ----------
        fit(X,y):
                X and y are array-like, represent features and labels.
                call fit() method to train Naive Bayes classifier.
        
        predict(X):
                
	
	"""
	
	def __init__(self,alpha=1.0,fit_prior=True,class_prior=None):
                self.alpha = alpha
		self.fit_prior = fit_prior
		self.class_prior = class_prior
		self.classes = None
		self.conditional_prob = None
		
	
	def _calculate_feature_prob(self,feature):
                values = np.unique(feature)
		total_num = float(len(feature))
		value_prob = {}
		for v in values:
			value_prob[v] = (( np.sum(np.equal(feature,v)) + self.alpha ) /
                                         ( total_num + len(values)*self.alpha))
		return value_prob
		
		
	def fit(self,X,y):
		#TODO: check X,y
		
		self.classes = np.unique(y)
		#calculate class prior probabilities: P(y=ck)
		if self.class_prior == None:
                        class_num = len(self.classes)
			if not self.fit_prior:
				self.class_prior = [1.0/class_num for _ in range(class_num)]  #uniform prior
			else:
				self.class_prior = []
				sample_num = float(len(y))
				for c in self.classes:
					c_num = np.sum(np.equal(y,c))
					self.class_prior.append(
                                                (c_num+self.alpha)/(sample_num+class_num*self.alpha))
		
		#calculate Conditional Probability: P( xj | y=ck )
		self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
		for c in self.classes:
			self.conditional_prob[c] = {}
			for i in range(len(X[0])):  #for each feature
                                feature = X[np.equal(y,c)][:,i]
				self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
		return self


        #given values_prob {value0:0.2,value1:0.1,value3:0.3,.. } and target_value
	#return the probability of target_value
	def _get_xj_prob(self,values_prob,target_value):
                return values_prob[target_value]
	
        #predict a single sample based on (class_prior,conditional_prob)
	def _predict_single_sample(self,x):
                label = -1
                max_posterior_prob = 0
                
		#for each category, calculate its posterior probability: class_prior * conditional_prob
                for c_index in range(len(self.classes)):
                        current_class_prior = self.class_prior[c_index]
			current_conditional_prob = 1.0
			feature_prob = self.conditional_prob[self.classes[c_index]]
			j = 0
			for feature_i in feature_prob.keys():
                                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i],x[j])
				j += 1
			
			#compare posterior probability and update max_posterior_prob, label
			if current_class_prior * current_conditional_prob > max_posterior_prob:
				max_posterior_prob = current_class_prior * current_conditional_prob
				label = self.classes[c_index]
		return label
	
	#predict samples (also single sample)			
	def predict(self,X):
                #TODO1:check and raise NoFitError 
                #ToDO2:check X
                if X.ndim == 1:
                        return self._predict_single_sample(X)
                else:
                        #classify each sample 	
                        labels = []
                        for i in range(X.shape[0]):
                                label = self._predict_single_sample(X[i])
                                labels.append(label)
                        return labels



#GaussianNB differ from MultinomialNB in these two method:
# _calculate_feature_prob, _get_xj_prob
class GaussianNB(MultinomialNB):
        """
        GaussianNB inherit from MultinomialNB,so it has self.alpha
        and self.fit() use alpha to calculate class_prior
        However,GaussianNB should calculate class_prior without alpha.
        Anyway,it make no big different

        """
        #calculate mean(mu) and standard deviation(sigma) of the given feature
        def _calculate_feature_prob(self,feature):
                mu = np.mean(feature)
                sigma = np.std(feature)
                return (mu,sigma)
        
        #the probability density for the Gaussian distribution 
        def _prob_gaussian(self,mu,sigma,x):
                return ( 1.0/(sigma * np.sqrt(2 * np.pi)) *
                        np.exp( - (x - mu)**2 / (2 * sigma**2)) )
        
        #given mu and sigma , return Gaussian distribution probability for target_value
        def _get_xj_prob(self,mu_sigma,target_value):
                return self._prob_gaussian(mu_sigma[0],mu_sigma[1],target_value)
