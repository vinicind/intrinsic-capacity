
from sklearn.metrics import r2_score
import copy
import operator
import random
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
import sys

import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model

from tensorflow.keras.layers import Flatten,Dense,Input,Reshape
from tensorflow.keras.models import Sequential
from random import randrange


class architecture:
    def __init__(self, neurons=None, activation=None, optimizer=None):
        self.neurons = neurons
        self.activation = activation
        self.optimizer = optimizer


class GA_assymetric_autoencoder:
    def __init__(self, shape, coding_size, X_train, X_test, DNA_parameter, epochs):
        self.shape = shape
        self.code_size = coding_size
        self.X_train = X_train
        self.X_test = X_test
        self.activations = DNA_parameter[0]
        self.optimizers = DNA_parameter[1]
        self.epochs = epochs
        
    
    #This one shows the real numbers of neurons instead of percentage
    
    def build_autoencoder(self, architecture, output='decoder'):
        decoder_arch = architecture['decoder']
        encoder_arch = architecture['encoder']
        code = architecture['code'][0]
        
        
        input_layer = Input(shape=encoder_arch[0].neurons, name='input')
        encoder_layer = input_layer
        arch_size = len(architecture)
        # Encoder layers
        for unit in encoder_arch[1:]:
            encoder_layer = Dense(unit.neurons, activation=unit.activation)(encoder_layer)

        # Bottleneck layer
        encoder_output = Dense(code.neurons, code.activation, name="bottleneck")(encoder_layer)

        decoder_layer = encoder_output

        # Decoder layers
        for unit in decoder_arch[:-1]:
            decoder_layer = Dense(unit.neurons, activation=unit.activation)(decoder_layer)

        output_layer = Dense(decoder_arch[-1].neurons, decoder_arch[-1].activation)(decoder_layer)

        # Define autoencoder model
        if output ==  'decoder':
            autoencoder = Model(input_layer, output_layer)
        elif output ==  'encoder':
            autoencoder = Model(input_layer, encoder_output)
        # Compile the model
        autoencoder.compile(optimizer=decoder_arch[-1].optimizer, loss='mean_squared_error')

        return autoencoder
    
    def print_architecture(self, architecture):
        archs = architecture['encoder'] + architecture['code'] + architecture['decoder'] 
        print('-'*30)
        for arch in archs:
            print(arch.neurons, arch.activation)
        print('optimizer', arch.optimizer)
        print('-'*30)
        
    def train(self, model):
        model.fit(self.X_train, self.X_train, 
            epochs= self.epochs, 
            verbose=0,
            batch_size=32,
            shuffle=True)
        decoded_data = model.predict(self.X_test)
        global_r2 = r2_score(self.X_test, decoded_data)
        return global_r2, model
    
    def create_population(
        self, 
        hidden_layers_encoder, 
        hidden_layers_decoder, 
        population_size=20, 
        max_dim_increase=60
        ):
        architectures=[]

        # unfold DNA_parameters:


        for pop in range(population_size):

            the_architecture_encoder = []
            the_architecture_decoder = []
            
            neurons_encoder = [max_dim_increase]
            neurons_decoder = [self.shape]
            #We only need to create and optimize architecture for half of the depth of the network because we want to create symmetric autoencoders
            the_architecture_encoder.append(architecture(neurons=self.shape)) # Input
            
            for i in range(hidden_layers_encoder):
                activation  = np.random.choice(self.activations)
                ch_neuron =  np.random.rand()
                units = round(neurons_encoder[i]*ch_neuron)
                if units==0 or units<=self.code_size:
                    break
                neurons_encoder.append(units)
                the_architecture_encoder.append(architecture(neurons=units,activation=activation)) # Encoder
                
            code_layer = architecture(neurons=self.code_size, activation=np.random.choice(self.activations)) # Latent dimension
            
            
            decoder_list = []
            for i in range(hidden_layers_decoder):
                activation  = np.random.choice(self.activations)
                ch_neuron =  np.random.rand()
                units = round(neurons_decoder[i]*ch_neuron)
                if units==0 or units<=self.code_size:
                    break
                neurons_decoder.append(units)
                decoder_list.append(architecture(neurons=units,activation=activation))
            
            the_architecture_decoder = decoder_list[::-1]
            the_architecture_decoder.append(architecture(neurons=self.shape, activation=np.random.choice(self.activations), optimizer=np.random.choice(self.optimizers)))
            architectures.append({
            'encoder': the_architecture_encoder,
            'code': [code_layer],
            'decoder': the_architecture_decoder
            })
        return architectures
    
    
    def _valid_encoder(self, encoder, min_neurons):
        # Verifica se o encoder é válido: decresce até a última camada
        return all(
            (encoder[i].neurons > encoder[i+1].neurons)
            and encoder[i].neurons > min_neurons and encoder[i+1].neurons > min_neurons
            for i in range(len(encoder) - 1)
        )

    def _valid_decoder(self, decoder, max_neurons, min_neurons):
        # Verifica se o decoder é válido: cresce até a última camada
        return all(
            (decoder[i].neurons < decoder[i+1].neurons)
            and decoder[i].neurons > min_neurons and decoder[i+1].neurons > min_neurons
            and decoder[i].neurons < max_neurons and decoder[i+1].neurons < max_neurons
            for i in range(len(decoder) - 1)
        )

    def recombine(self, ae1, ae2):
        encoder1 = ae1['encoder'][1:]
        encoder2 = ae2['encoder'][1:]
        decoder1 = ae1['decoder'][:-1]
        decoder2 = ae2['decoder'][:-1]
        # Selecionar pontos de corte para o encoder e o decoder
        if len(encoder1) == 0:
            cut_point_encoder1 = 0
        else:
            cut_point_encoder1 = np.random.randint(0, len(encoder1)+1)
        
        if len(encoder2) == 0:
            cut_point_encoder2 = 0
        else:
            cut_point_encoder2 = np.random.randint(0, len(encoder2)+1)
        
        if len(decoder1) == 0:
            cut_point_decoder1 = 0 
        else:
            cut_point_decoder1 = np.random.randint(0, len(decoder1)+1)
            
        if len(decoder2) == 0:
            cut_point_decoder2 = 0
        else:
            cut_point_decoder2 = np.random.randint(0, len(decoder2)+1)
        
        # Recombinar os encoders e decoders

        new_encoder = encoder1[:cut_point_encoder1] + encoder2[cut_point_encoder2:]
        new_decoder = decoder2[:cut_point_decoder2] + decoder1[cut_point_decoder1:]

        # Verificar e corrigir se necessário
        if not self._valid_encoder(new_encoder, self.code_size):
            new_encoder = sorted(new_encoder, key=lambda x: x.neurons, reverse=True)
        if not self._valid_decoder(new_decoder, self.shape, self.code_size):
            new_decoder = sorted(new_decoder, key=lambda x: x.neurons)

        # Retornar os novos encoder e decoder recombinados
        return {
            'encoder': [ae1['encoder'][0]] + new_encoder,
            'code': ae1['code'],
            'decoder':   new_decoder + [ae1['decoder'][-1]]
        }
        
    def mutate(self, ae):
        while True:
            # Gerar fator de mutação
            encoder = ae['encoder'][1:]
        
            decoder = ae['decoder'][:-1]
            mutation_factor_encoder = np.random.rand() + np.random.randint(0, 2)
            mutation_factor_decoder = np.random.rand() + np.random.randint(0, 2)
            
            new_encoder = []
            new_decoder = []
            
            # Aplicar mutação
            for layer in encoder:
                new_layer = copy.copy(layer)
                new_layer.neurons = round(layer.neurons * mutation_factor_encoder)
                new_layer.activation = np.random.choice(self.activations)
                new_encoder.append(new_layer)
            for layer in decoder:
                new_layer = copy.copy(layer)
                new_layer.neurons = round(layer.neurons * mutation_factor_decoder)
                new_layer.activation = np.random.choice(self.activations)
                new_decoder.append(new_layer)

            # Verificar se o encoder é válido
            if not self._valid_encoder(new_encoder, self.code_size):
                continue
            
            # Verificar se o decoder é válido
            if not self._valid_decoder(new_decoder, self.shape, self.code_size):
                continue
            
            ae['decoder'][-1].optimizer = np.random.choice(self.optimizers)
            
            # Se ambos são válidos, retornar os mutados
            return {
                'encoder': [ae['encoder'][0]] + new_encoder,
                'code': ae['code'],
                'decoder':   new_decoder + [ae['decoder'][-1]]
            }
        
    def GA(self, population, n_generations = 100, mutation_rate=0.2,Crossover=True,Mutation=True):

        best_architecture = []
        pop_size = len(population)

        for i in range(n_generations):
            print("Generation:",i)

            sel_index = random.sample([i for i in range(pop_size)], 3)
            
            


            P1= population[sel_index[0]]
            P2= population[sel_index[1]]
            P3= population[sel_index[2]]

            
            P1_model = self.build_autoencoder(P1)
            P2_model = self.build_autoencoder(P2)
            P3_model = self.build_autoencoder(P3)

            P1_model_trained = self.train(P1_model)
            P2_model_trained = self.train(P2_model)
            P3_model_trained = self.train(P3_model)
            
            # Collect all fitness scores with corresponding model identifiers
            fitness_scores = [
                {
                    'p': P1,
                    'i': sel_index[0],
                    's': P1_model_trained[0],
                    'm': P1_model_trained[1]
                },
                {
                    'p': P2,
                    'i': sel_index[1],
                    's': P2_model_trained[0],
                    'm': P2_model_trained[1]
                },
                {
                    'p': P3,
                    'i': sel_index[2],
                    's': P3_model_trained[0],
                    'm': P3_model_trained[1]
                }
            ]

            # Sort models based on fitness scores in descending order (higher is better, adjust if necessary)
            sorted_fitness = sorted(fitness_scores, key=lambda x: x['s'], reverse=True)

            # Get best, second best, and worst performing models
            best = sorted_fitness[0]['p']  # Best
            second_best =  sorted_fitness[1]['p']  # Second best
            worst =  sorted_fitness[2]['p']  # Worst


                
            print("Best: ")
            self.print_architecture(best)
        
            print("Second Best: ")
            self.print_architecture(second_best)
            
            print("Loser: ")
            self.print_architecture(worst)
            
            
            
            if Crossover:
                best_child = self.recombine(best, second_best)
                second_best_child = self.recombine(best, worst)
            
            print("Best Child: ")
            self.print_architecture(best_child)
            
                    
            print("Second Best Child: ")
            self.print_architecture(second_best_child)
            
            
            if Mutation:
                if np.random.rand() < mutation_rate:  
                    second_best_child = self.mutate(second_best_child)    
                    print("Second Best Child Mutate: ")
                    self.print_architecture(second_best_child)
                
        
            

            best_architecture.append(sorted_fitness[0])
            print('Best result in generation', sorted_fitness[0]['s'])

            population[sorted_fitness[1]['i']] = best_child
            population[sorted_fitness[2]['i']] = second_best_child

        self.save_best_model_and_architecture(best_architecture)
        return best_architecture
    
    def save_best_model_and_architecture(self, results):
    # Find the result with the best score
        best_result = max(results, key=lambda x: x['s'])
        print(best_result)
        # Save the Keras model
        model = best_result['m']
        model.save('./out/best_model.h5')
        
        # Save the architecture details in a text file
        architecture = best_result['p']
        with open('./out/architecture.txt', 'w') as f:
            # Redirect stdout to the file
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                self.print_architecture(architecture)
            finally:
                # Restore stdout
                sys.stdout = original_stdout