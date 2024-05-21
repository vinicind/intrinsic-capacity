import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from genetic_algorithm import GA_assymetric_autoencoder
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model

from tensorflow.keras.layers import Flatten,Dense,Input,Reshape
from tensorflow.keras.models import Sequential
from random import randrange


df = pd.read_csv('Data/processed_df.csv')


ci_map = {
    'cognitive': ['semantic_memory', 'verbal_fluency', 'memory_recall', 'temporal_orientation'],
    'psychologicao': ['depression_scale', 'sleep_quality'],
    'locomotor': ['gait_speed', 'balance'],
    'vitality': ['handgrip', 'poor_endurance', 'weight_loss', 'exaustion'],
    'sensory': ['distance_vision', 'near_vision', 'hearing_deficit']
}

# ci_map = {
#     'cognitive': ['semantic_memory', 'verbal_fluency', 'memory_recall'],
#     'psychologicao': ['depression_scale', 'sleep_quality'],
#     'locomotor': ['gait_speed', 'balance'],
#     'vitality': ['handgrip', 'poor_endurance', 'weight_loss', 'exaustion'],
#     'sensory': ['distance_vision', 'near_vision', 'hearing_deficit']
# }
ci_cols = []
[ci_cols.extend(value) for value in ci_map.values()]

df_clean = df.dropna(subset=ci_cols)
df_clean.shape



# Define dataset
df_autoencoder = df_clean[ci_cols]
# Split the data into training and testing sets
X_train, X_test = train_test_split(df_autoencoder, test_size=0.2, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=42)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape


# DNA[0] = activations
# DNA[1] = optimizer
DNA_parameter = [["tanh","softmax","relu","sigmoid","linear"],
                 ["sgd","adam"]]


ga = GA_assymetric_autoencoder(
    shape=X_train.shape[1],
    coding_size=2, 
    X_train=X_train_scaled, 
    X_test=X_val_scaled, 
    DNA_parameter=DNA_parameter, 
    epochs=50
)


archs = ga.create_population(hidden_layers_encoder=4, hidden_layers_decoder=3)
results = ga.GA(population=archs, n_generations=100)


best_result = max(results, key=lambda x: x['s'])
print(best_result)