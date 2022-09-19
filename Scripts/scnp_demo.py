from losses import *
from data_preprocessing import *
#from model_builder import *
import evidential_deep_learning as edl
import sklearn.model_selection
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import keras_tuner as kt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


## Importing csv data file from GitHub repository
data = pd.read_csv('https://raw.githubusercontent.com/GormleyLab/Data-Driven-Design-of-SCNPs/main/Data/SCNP_DLS_SAXS_Data.csv')

## Removing measured parameters that are not porod exponent
data.drop(labels=['Sample','Rh (nm)','Rg GNOM (nm)','Dmax GNOM','Rg BIFT (nm)','Dmax BIFT','Porod Volume (A^3)','Rg/Rh'],axis=1,inplace=True)

#Scaling data utilizing scikit's RobustScaler function.
transformed_features,transformed_labels = transform_fit(data)

# Creating a validation 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(transformed_features, transformed_labels, test_size=0.20,random_state=52)

#Defining our neural network with 4 densely connected layers and one evidential regression layer for the output.
model = tf.keras.Sequential([
tf.keras.layers.Dense(50, activation='relu'),
tf.keras.layers.Dense(units=96, activation='relu'),
tf.keras.layers.Dense(units=96, activation='relu'),
tf.keras.layers.Dense(50, activation='relu'),
edl.layers.DenseNormalGamma(1)
])

# compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=.0002)
model.compile(optimizer=optimizer, loss=EvidentialRegression) #loss is defined in the losses.py file
model.fit(X_train, y_train, epochs=50, batch_size=128)

#performing predictions on the left-out dataset.
predictions = model.predict(X_test)


# As the edl layer provides 4 outputs, we obtain the mean predictions and compare it to the true porod exponent values to evaluate performance.
mu, v, alpha, beta = tf.split(predictions, 4, axis=-1)
mu = mu[:,0]
y_test_r = y_test[:,0]



print(r2_score(y_test_r,mu))
