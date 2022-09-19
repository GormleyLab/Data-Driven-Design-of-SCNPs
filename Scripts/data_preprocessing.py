import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def transform_fit(data):

	features = np.array(data[data.columns[1:-1]])
	labels = np.array(data[data.columns[0]]).reshape(-1,1)

	transformer = RobustScaler()
	transformed_features = transformer.fit_transform(features)
	transformed_labels = transformer.fit_transform(labels)
	
	return transformed_features,transformed_labels

