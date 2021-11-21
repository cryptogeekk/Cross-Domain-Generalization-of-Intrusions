from tensorflow import keras

def get_model():
    model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[80,]),
    keras.layers.Dense(200,activation='tanh'),
    keras.layers.Dense(100,activation='tanh'),
    keras.layers.Dense(56,activation='sigmoid')
    ])
    
    return model

