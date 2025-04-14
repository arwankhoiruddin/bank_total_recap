import pandas as pd
import numpy as np
import ahpy as ahpy

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Load data

df = pd.read_csv('bank.csv')
np_bank = df.to_numpy()

"""Initialize parameters"""

criteria = ['HC',
            'FC',
            'GCG',
            'RM',
            'BP',
            'IDN',
            'GC',
            'IT',
            'DBM',
            'OA']

success_criteria = [
        'success',
        'failed'
    ]

weights = np.zeros((np_bank.shape[0], 2*len(criteria)))

"""Run AHP"""

cr = []
for data_index in range(0, np_bank.shape[0]):
    print(f'Data number {data_index}')

    data_sample = np_bank[data_index][:46]

    np_weight = np.zeros((len(criteria), len(criteria)))

    idx = 1
    leng = len(criteria) - 1

    for i in range(0, len(criteria)):
        if i < len(criteria):
            data = data_sample[idx: idx + leng]
            np_weight[i, (i+1):] = data
            idx = idx + leng
            leng = leng - 1
        for j in range(0, len(criteria)):
            if i == j:
                np_weight[i][j] = 1
            if i > j:
                np_weight[i][j] = 1 / np_weight[j][i]

    # Convert the numpy matrix into a dictionary of comparisons for ahpy
    comparison_dict = {}
    for i in range(len(criteria)):
        for j in range(i+1, len(criteria)):
            comparison_dict[(criteria[i], criteria[j])] = np_weight[i][j]

    # Initialize the ahpy Compare object
    ahp_compare = ahpy.Compare(name='AHP_Model', comparisons=comparison_dict, precision=3)

    # Print the priority vector (weights)

    target_weights = ahp_compare.target_weights
    print("Criteria Weights:")
    print(target_weights)

    # Print the consistency ratio
    print("Consistency Ratio:")
    print(ahp_compare.consistency_ratio)
    cr.append(ahp_compare.consistency_ratio)

    temp = np.zeros((1, len(criteria)))

    for j in range(0, len(criteria)):
        crit = target_weights[criteria[j]]
        temp[0, j] = crit

    # put criteria matrix as input
    weights[data_index][:10] = temp

    data_sample_success = np_bank[data_index][46:56]

    param_matrix = np.zeros((10, 2, 2))

    success = []
    data_idx = 0
    for data in data_sample_success:
        for i in range(0, 2):
            for j in range(0, 2):
                if i == j:
                    param_matrix[data_idx][i][j] = 1
        param_matrix[data_idx][0][1] = data_sample_success[data_idx]
        param_matrix[data_idx][1][0] = 1 / float(data_sample_success[data_idx])

        # Convert the numpy matrix into a dictionary of comparisons for ahpy
        comparison_dict = {}
        for i in range(0, 2):
            for j in range(0, 2):
                comparison_dict[(success_criteria[i], success_criteria[j])] = param_matrix[data_idx][i][j]

        # Initialize the ahpy Compare object
        ahp_compare = ahpy.Compare(name='AHP_Model', comparisons=comparison_dict, precision=3)

        # Print the priority vector (weights)
        print(f'Parameter: {criteria[data_idx]}')

        target_weights_success = ahp_compare.target_weights
        print("Criteria Weights:")
        print(target_weights_success)
        success.append(target_weights_success['success'])
        print()
        data_idx += 1

    # put success rate as output
    weights[data_index][10:20] = success

"""Run ANN"""

weight_df = pd.DataFrame(weights)
weight_df['cr'] = cr
weight_df.to_csv('weight.csv', index=False)

"""Define Inputs and Outputs"""

X_train, X_test, Y_train, Y_test = train_test_split(weights[:, :10], weights[:, 10:], test_size=0.3, random_state=42)

np_ytrain = np.array(Y_train)
np_ytest = np.array(Y_test)

squared_ytrain = np_ytrain ** 3
squared_ytest = np_ytest ** 3

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train = scaler.fit_transform(X_train)

# Only transform the test data
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='softmax')) # relu
model.add(Dropout(0.2))  # Dropout layer to reduce overfitting
model.add(Dense(10, activation='sigmoid'))

# Compile the model (categorical crossentropy for multi-class classification)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=10, validation_data=(X_test, Y_test))

HC = 0.212
FC = 0.168
GCG = 0.084
RM = 0.126
BP = 0.048
IDN = 0.044
GC = 0.029
IT = 0.122
DBM = 0.042
OA = 0.126

input_data = np.array([HC, FC, GCG, RM, BP, IDN, GC, IT, DBM, OA])

sample_input = weights[1, :10]
sample_output = weights[1, 10:]

prediction_result = model.predict(input_data.reshape(1, -1))

print([round(num, 2) for num in prediction_result[0]])
print(sample_output)

