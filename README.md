# metas-learn
Collection of machine learning algorithms implemented in Python as part of a project funded by the Swiss Innovation Agency (36779.1 IP-ENG), read more about it here: https://www.aramis.admin.ch/Grunddaten/?ProjectID=44523.

The algorithms have been implemented as standalone classes; this leads to redundancy and more code because the same functions are present in several files, but it easier to modify or implement them as part of A.I. solutions.

The **regression** and **classification** algorithms are based on **stochastic gradient descent** (and the clustering algorithm is based on expectation-maximization). Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function, e.g. **mean squared error** in regression or **binary cross-entropy** in classification. Gradient descent is based on the observation that if the multi-variable function C(W) is defined and differentiable in a neighborhood of a point W, then C(W) decreases fastest if one goes from W in the direction of the negative gradient of C at W, -dC/dW. 

![gradient](Figures/gradient.png)

However, by including only a randomly selected subset of data in the computation of the gradient, the algorithm becomes stochastic since it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data).


## Examples

### Regression

#### Load libraries.
  	from StandardScaler import StandardScaler
	from NeuralNetworkRegressor import NeuralNetworkRegressor
	import numpy as np
	from sklearn.datasets import load_boston
	import matplotlib.pyplot as plt
  
#### Load data.
As a guidline, it is recommended to scale data before training, e.g. using a standard scaling (zero mean, unit variance).

  	scaler = StandardScaler()
  	X, y = load_boston(return_X_y=True)
  	X, y = scaler.fit_transform(X), scaler.fit_transform(y.reshape(-1, 1))
  	nnr = NeuralNetworkRegressor(hidden_layer_sizes=(10,), alpha=1.0)
  
#### Fit model.
	nnr.fit(X, y)
	y_pred = nnr.predict(X)
	
#### Evaluate model.

##### Learning Curve
	
![loss](Figures/loss_NN.png)

##### Agreement
	
![congruency](Figures/congruency_NN.png)

### Classification

#### Load libraries.
  	from StandardScaler import StandardScaler
	from LogisticClassifier import LogisticClassifier
	from sklearn.metrics import precision_recall_curve
	import numpy as np
	from sklearn.datasets import load_breast_cancer
	import matplotlib.pyplot as plt
  
#### Load data.
As a guidline, it is recommended to scale data before training, e.g. using a standard scaling (zero mean, unit variance).

  	scaler = StandardScaler()
  	X, y = load_breast_cancer(return_X_y=True)
  	X = scaler.fit_transform(X)
  	lc = LogisticClassifier(alpha=1.0)
  
#### Fit model.
	lc.fit(X, y)
	y_pred = lc.predict(X)
	
#### Evaluate model.

##### Learning Curve
	
![loss](Figures/loss_LC.png)

##### Precision-Recall Curve
	    
![congruency](Figures/congruency_LC.png)
