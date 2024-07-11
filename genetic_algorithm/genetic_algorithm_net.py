
from sklearn.metrics import r2_score
import copy
import operator
from tensorflow.keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
import random
import pandas as pd
import keras.metrics as metrics
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

from tensorflow.keras.layers import Flatten,Dense,Input,Reshape
from tensorflow.keras.models import Sequential
from random import randrange


class architecture:
    def __init__(self, neurons=None, activation=None, optimizer=None):
        self.neurons = neurons
        self.activation = activation
        self.optimizer = optimizer


class GA:
    def __init__(self, coding_size, X_train, X_test, y_train, y_test, DNA_parameter, epochs):
        self.shape = X_train.shape[1]
        self.target_shape = len(y_train.unique())
        self.code_size = coding_size
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.activations = DNA_parameter[0]
        self.optimizers = DNA_parameter[1]
        self.epochs = epochs
        
    
    #This one shows the real numbers of neurons instead of percentage
    
    def build_neural_net(self, architecture, output='decoder'):
        decoder_arch = architecture['decoder']
        encoder_arch = architecture['encoder']
        code = architecture['code'][0]
        
        
        input_layer = Input((encoder_arch[0].neurons, ), name='input')
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
        
        loss = BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
        autoencoder.compile(optimizer=decoder_arch[-1].optimizer, loss=loss, metrics=[
            'accuracy',
            metrics.Recall(name='recall'),
            metrics.Precision(name='precision')
        ])

        return autoencoder
    
    def print_architecture(self, architecture):
        archs = architecture['encoder'] + architecture['code'] + architecture['decoder'] 
        print('-'*30)
        for arch in archs:
            print(arch.neurons, arch.activation)
        print('optimizer', arch.optimizer)
        print('-'*30)
        
    def train(self, model):
        model.fit(self.X_train, self.y_train, 
            epochs= self.epochs, 
            verbose=0,
            batch_size=32, validation_split=0.2)
        
        # Evaluate the model
        loss, accuracy, precision, recall = model.evaluate(self.X_test, self.y_test)
        if precision > 0 and recall > 0:
            f1 = (2*precision*recall)/(precision+recall)
        else:
            f1 = 0
        return accuracy, f1, precision, recall, model
    
    def create_population(
        self, 
        hidden_layers_encoder, 
        hidden_layers_decoder, 
        max_dim_increase,
        population_size=20, 
        ):
        architectures=[]

        # unfold DNA_parameters:


        for pop in range(population_size):

            the_architecture_encoder = []
            the_architecture_decoder = []
            
            neurons_encoder = [max_dim_increase]
    
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
            

        
            the_architecture_decoder.append(architecture(neurons=1, activation='sigmoid', optimizer=np.random.choice(self.optimizers)))
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
        # Selecionar pontos de corte para o encoder e o decoder
        if len(encoder1) == 0:
            cut_point_encoder1 = 0
        else:
            cut_point_encoder1 = np.random.randint(0, len(encoder1)+1)
        
        if len(encoder2) == 0:
            cut_point_encoder2 = 0
        else:
            cut_point_encoder2 = np.random.randint(0, len(encoder2)+1)
        # Recombinar os encoders e decoders

        new_encoder = encoder1[:cut_point_encoder1] + encoder2[cut_point_encoder2:]
        # Verificar e corrigir se necessário
        if not self._valid_encoder(new_encoder, self.code_size):
            new_encoder = sorted(new_encoder, key=lambda x: x.neurons, reverse=True)
        # Retornar os novos encoder e decoder recombinados
        return {
            'encoder': [ae1['encoder'][0]] + new_encoder,
            'code': ae1['code'],
            'decoder':  ae1['decoder']
        }
        
    def mutate(self, ae, max_try=100):
        while max_try:
            # Gerar fator de mutação
            max_try -= 1
            encoder = ae['encoder'][1:]
            
            
            if len(encoder) == 0:
                return {
                    'encoder': [ae['encoder'][0]],
                    'code': ae['code'],
                    'decoder':  ae['decoder']
                }

            mutation_factor_encoder = np.random.rand() + np.random.randint(0, 2)
            
            new_encoder = copy.deepcopy(encoder)
            
            new_layer = np.random.choice(new_encoder)
            new_layer.neurons = round(new_layer.neurons * mutation_factor_encoder)
            new_layer.activation = np.random.choice(self.activations)
            
        
            # Verificar se o encoder é válido
            if not self._valid_encoder(new_encoder, self.code_size):
                continue
            

            ae['decoder'][-1].optimizer = np.random.choice(self.optimizers)
            
            # Se ambos são válidos, retornar os mutados
            return {
                'encoder': [ae['encoder'][0]] + new_encoder,
                'code': ae['code'],
                'decoder':  ae['decoder']
            }
        return ae
        
    def GA(self, population, n_generations = 100, mutation_rate=0.2,Crossover=True,Mutation=True):

        best_architecture = []
        pop_size = len(population)

        for i in range(n_generations):
            print("Generation:",i)

            sel_index = random.sample([i for i in range(pop_size)], 3)
            
            


            P1= population[sel_index[0]]
            P2= population[sel_index[1]]
            P3= population[sel_index[2]]

            
            P1_model = self.build_neural_net(P1)
            P2_model = self.build_neural_net(P2)
            P3_model = self.build_neural_net(P3)

            P1_model_trained = self.train(P1_model)
            P2_model_trained = self.train(P2_model)
            P3_model_trained = self.train(P3_model)
            
            # Collect all fitness scores with corresponding model identifiers
            fitness_scores = [
                {
                    'p': P1,
                    'i': sel_index[0],
                    'acc': P1_model_trained[0],
                    'f1': P1_model_trained[1],
                    'precision': P1_model_trained[2],
                    'recall': P1_model_trained[3],
                    'm': P1_model_trained[4]
                },
                {
                    'p': P2,
                    'i': sel_index[1],
                    'acc': P2_model_trained[0],
                    'f1': P2_model_trained[1],
                    'precision': P2_model_trained[2],
                    'recall': P2_model_trained[3],
                    'm': P2_model_trained[4]
                },
                {
                    'p': P3,
                    'i': sel_index[2],
                    'acc': P3_model_trained[0],
                    'f1': P3_model_trained[1],
                    'precision': P3_model_trained[2],
                    'recall': P3_model_trained[3],
                    'm': P3_model_trained[4]
                }
            ]

            # Sort models based on fitness scores in descending order (higher is better, adjust if necessary)
            sorted_fitness = sorted(fitness_scores, key=lambda x: x['f1'], reverse=True)

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
                    print('Mutating...')
                    second_best_child = self.mutate(second_best_child)    
                    print("Second Best Child Mutate: ")
                    self.print_architecture(second_best_child)
                
        
            

            best_architecture.append(sorted_fitness[0])
            print('Best result in generation (f1)', sorted_fitness[0]['f1'] )
            print('Best result in generation (acc)',  sorted_fitness[0]['acc'])
            
            
           

            population[sorted_fitness[1]['i']] = best_child
            population[sorted_fitness[2]['i']] = second_best_child

        self.save_best_model_and_architecture(best_architecture)
        return best_architecture
    
    def save_best_model_and_architecture(self, results):
    # Find the result with the best score
        best_result = max(results, key=lambda x: x['f1'])
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
                print('Best result in generation (f1)', best_result['f1'] )
                print('Best result in generation (acc)',  best_result['acc'])
            
            finally:
                # Restore stdout
                sys.stdout = original_stdout