from Losses import *
from data_preprocessing import *
import sklearn.model_selection
import tensorflow as tf
import evidential_deep_learning as edl
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

r2_container = []
trial_container = []

def model_wrapper(X_train, X_test, y_train, y_test):
  


  def model(hp):

      hp_units = hp.Int('units', min_value=96, max_value=96, step=32)
      hp_units2 = hp.Int('units2', min_value=96, max_value=96, step=32)
      hp_learning_rate = hp.Choice('learning_rate', values=[0.0002])

      def EvidentialRegressionLoss(true, pred):
        return EvidentialRegression(true, pred, coeff=.1)

      model = tf.keras.Sequential([
      tf.keras.layers.Dense(50, activation='relu'),
      tf.keras.layers.Dense(units=hp_units, activation='relu'),
      tf.keras.layers.Dense(units=hp_units2, activation='relu'),
      tf.keras.layers.Dense(50, activation='relu'),
      edl.layers.DenseNormalGamma(1)
      ])

      # compile the model
      optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
      model.compile(optimizer=optimizer, loss=EvidentialRegression) #loss=EvidentialRegressionLoss

      return model

  class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=128, epochs=1):
      global r2_container,trial_container
      cv = KFold(2,shuffle=True)
      val_losses = []
      r2_losses = []
      for train_indices, test_indices in cv.split(x):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model = self.hypermodel.build(trial.hyperparameters)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(x_test, y_test),callbacks=[stop_early])
        val_losses.append(model.evaluate(x_test, y_test))
        temp_preds = model(x_test)
        mu, v, alpha, beta = tf.split(temp_preds, 4, axis=-1)
        mu = mu[:,0]
        y_test_r = y_test[:,0]
        r2_container.append(r2_score(y_test_r,mu))
        r2_losses.append(r2_score(y_test_r,mu))
        trial_container.append(trial.trial_id)

        #return super(CVTuner, self).run_trial(trial, x, y, batch_size=128, epochs=1)

        print(val_losses,r2_losses)
      #self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
      #self.save_model(trial.trial_id, model)

  tuner = CVTuner(
    hypermodel=model,
    oracle=kt.oracles.BayesianOptimization(
      objective='val_loss',
      max_trials=5))

  tuner.search(X_train, y_train, epochs=3,batch_size=128,validation_data=(X_test, y_test))

  return tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = model_wrapper(X_train, X_test, y_train, y_test)