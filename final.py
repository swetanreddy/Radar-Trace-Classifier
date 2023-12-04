import numpy as np
from sklearn.naive_bayes import GaussianNB

def fill_nas(train_data, test_data):
    # Replace NaNs 
    for data in [train_data, test_data]:
        for i in range(data.shape[0]):
            na_idx = np.isnan(data[i])
            data[i, na_idx] = np.nanmean(data[i])
            
            
def extract_features(data, labels, bird_likelihoods, plane_likelihoods):
    
    features = []
    
    for trace in data:
      
        # Calculate properties like mean, variance etc
        mean = np.mean(trace)
        var = np.var(trace)
        
        # Get likelihood based on speed 
        speed = np.abs(trace[0] - trace[-1]) / len(trace)
        b_likelihood = bird_likelihoods[round(speed)] 
        p_likelihood = plane_likelihoods[round(speed)]
        
        # Add features
        features.append([mean, var, b_likelihood, p_likelihood])
        
    return np.array(features)

# Load likelihoods
likelihoods = np.loadtxt('likelihood.txt')
bird_likelihoods = likelihoods[0] 
plane_likelihoods = likelihoods[1]

# Load data
train_data = np.loadtxt('training.txt')
train_labels = np.concatenate((np.zeros(10), np.ones(10)))

test_data = np.loadtxt('testing.txt')

# Replace NaNs 
fill_nas(train_data, test_data)

# Extract features including likelihoods
train_X = extract_features(train_data, train_labels, bird_likelihoods, plane_likelihoods)
test_X = extract_features(test_data, None, bird_likelihoods, plane_likelihoods)  

# Train model
model = GaussianNB()
model.fit(train_X, train_labels)

# Initial class priors
p_b = 0.5  
p_a = 0.5

prev_label = None
for x in test_X:
  
    # Compute predictions
    probs = model.predict_proba([x])[0]
    pred = model.predict([x])[0]
    
    # Apply transition conditions
    if prev_label == 'bird' and pred == 0:  
        if np.random.rand() <= 0.9:
           pred = 0
            
    if prev_label == 'airplane' and pred == 1:
       if np.random.rand() <= 0.9:
           pred = 1
           
    # Print strings
    if pred == 0:
        print('bird')
        prev_label = 'bird'
    else:
        print('plane')
        prev_label = 'airplane'