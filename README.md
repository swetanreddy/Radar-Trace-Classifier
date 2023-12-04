# Radar-Trace-Classifier

## Problem Statement

Given a time-series of sensor data, classify whether it corresponds to a bird or airplane. The data has missing values and transitions between classes needs to be handled smoothly.

## Data

The data is in two files:

**training.txt** - 20 tracks (10 bird, 10 plane). Each sample has a 600 length time-series.

**testing.txt** - 10 unlabeled time-series tracks to predict

## Approach

A Gaussian Naive Bayes classifier is used with additional components for:

- Missing value imputation
- Feature extraction including speed-based likelihoods
- Smoothing model transitions

### Missing Value Imputation

The data has missing values indicated by NaNs. A mean imputation strategy is used to fill these missing values by the mean of that feature column. This simplifies modeling without needing to handle NaNs.

```python
def fill_nas(train_data, test_data):
    for data in [train_data, test_data]:
        for i in range(data.shape[0]):
            na_idx = np.isnan(data[i])
            data[i, na_idx] = np.nanmean(data[i])  
```

### Feature Extraction

Domain specific features are extracted including:

- Mean, variance: Descriptive stats of time-series
- Speed likelihood: Using speed of object, lookup likelihoods from provided table `likelihood.txt`. This helps discriminate between typically slow birds and fast planes.

```python  
def extract_features(data, labels, bird_likelihoods, plane_likelihoods):
      
    for trace in data:
      
        mean = np.mean(trace) 
        var = np.var(trace)
        
        speed = np.abs(trace[0] - trace[-1]) / len(trace) 
        b_likelihood = bird_likelihoods[round(speed)]  
        p_likelihood = plane_likelihoods[round(speed)]
        
        features.append([mean, var, b_likelihood, p_likelihood])
```

### Gaussian Naive Bayes Classification

A GaussianNB classifier is trained on the extracted features. This assumes independence between features and models each feature as a Gaussian distribution per class.

```python
model = GaussianNB()
model.fit(train_X, train_labels) 
model.predict(test_X)
```

### Smoothing Model Transitions

To avoid rapid toggling of predictions, transition probabilities are controlled between classes. If previous prediction was bird, probability of predicting bird again is increased.

```python
if prev_label == 'bird' and pred == 0:  
    if np.random.rand() <= 0.9:
       pred = 0       
```

## Usage

To run:

```
python final.py
```

## Future Improvements

Some ways to improve the model:

- Tuning the transition smoothing thresholds
- Adding more descriptive features
- Using RNN/LSTM to model time-series more accurately
- Ensemble modeling to combine multiple approaches
