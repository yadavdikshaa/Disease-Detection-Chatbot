"""
Medical Chatbot - Meddy
Project for course Intelligent Systems at Faculty of Organization and Informatics Varaždin, University of Zagreb

(c) 2021 Alagić Aldin, Benkus Maja, Košmerl Igor, Krištofić Miro

This script contains:
1. Natural Language Processing (NLP) model training for symptom recognition
2. Disease prediction model training using various ML algorithms
"""

import json
import nltk
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from nnet import NeuralNet
from nltk_utils import bag_of_words
import matplotlib.pyplot as plt

# Download required NLTK data if not already present
print("Checking for NLTK data...")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
print("NLTK data ready!\n")

if __name__ == '__main__':
    # ============================================================================
    # PART 1: Natural Language Processing (NLP) Model Training
    # ============================================================================

    print("=" * 80)
    print("PART 1: NLP Model Training")
    print("=" * 80)

    # Load intents data
    with open('intents_short.json', 'r') as f:
        intents = json.load(f)

    # Prepare training data
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Synthetic dataset for validation
    xy_test = [
    (['ca', "n't", 'think', 'straight'], 'altered_sensorium'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['bloody', 'poop'], 'bloody_stool'),
    (['blurred', 'vision'], 'blurred_and_distorted_vision'),
    (['ca', "n't", 'breathe'], 'breathlessness'),
    (['Yellow', 'liquid', 'pimple'], 'yellow_crust_ooze'),
    (['lost', 'weight'], 'weight_loss'),
    (['side', 'weaker'], 'weakness_of_one_body_side'),
    (['watering', 'eyes'], 'watering_from_eyes'),
    (['brief', 'blindness'], 'visual_disturbances'),
    (['throat', 'hurts'], 'throat_irritation'),
    (['extremities', 'swelling'], 'swollen_extremeties'),
    (['swollen', 'lymph', 'nodes'], 'swelled_lymph_nodes'),
    (['dark', 'under', 'eyes'], 'sunken_eyes'),
    (['stomach', 'blood'], 'stomach_bleeding'),
    (['blood', 'urine'], 'spotting_urination'),
    (['sinuses', 'hurt'], 'sinus_pressure'),
    (['watery', 'from', 'nose'], 'runny_nose'),
    (['have', 'to', 'move'], 'restlessness'),
    (['red', 'patches', 'body'], 'red_spots_over_body'),
    (['sneeze'], 'continuous_sneezing'),
    (['coughing'], 'cough'),
    (['skin', 'patches'], 'dischromic_patches'),
    (['skin', 'bruised'], 'bruising'),
    (['burning', 'pee'], 'burning_micturition'),
    (['hurts', 'pee'], 'burning_micturition'),
    (['Burning', 'sensation'], 'burning_micturition'),
    (['chest', 'pressure'], 'chest_pain'),
    (['pain', 'butt'], 'pain_in_anal_region'),
    (['heart', 'bad', 'beat'], 'palpitations'),
    (['fart', 'lot'], 'passage_of_gases'),
    (['cough', 'phlegm'], 'phlegm'),
    (['lot', 'urine'], 'polyuria'),
    (['Veins', 'bigger'], 'prominent_veins_on_calf'),
    (['Veins', 'emphasized'], 'prominent_veins_on_calf'),
    (['yellow', 'pimples'], 'pus_filled_pimples'),
    (['red', 'nose'], 'red_sore_around_nose'),
    (['skin', 'yellow'], 'yellowish_skin'),
    (['eyes', 'yellow'], 'yellowing_of_eyes'),
    (['large', 'thyroid'], 'enlarged_thyroid'),
    (['really', 'hunger'], 'excessive_hunger'),
    (['always', 'hungry'], 'excessive_hunger'),
    ]

    # Process words
    stemmer = PorterStemmer()
    ignore_words = ['?', '!', '.', ',']
    all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Prepare training data
X_train = []
y_train = []

for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Prepare test data
X_test = []
y_test = []

for (pattern, tag) in xy_test:
    bag = bag_of_words(pattern, all_words)
    X_test.append(bag)
    label = tags.index(tag)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# ChatDataset class
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Neural network validation function
def nn_validation():
    dataset_train = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    loss_train = []
    loss_test = []

    for lr in learning_rates:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)

        print(f"lr: {lr}, train")
        for epoch in range(num_epochs):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device)

                outputs = model(words)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % (num_epochs / 2) == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

        print(f'final loss = {loss.item():.4f}')
        loss_train.append(loss.item())

        y_predicted = []
        for x in X_test:
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x)
            output = model(x)
            _, predicted = torch.max(output, dim=1)
            y_pred = predicted.item()
            y_predicted.append(y_pred)

        print("y_predicted:", y_predicted)
        y_predicted = np.array(y_predicted)
        loss_test.append(accuracy_score(y_test, y_predicted))
        print()

    return loss_train, loss_test

# Training parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rates = [0.01, 0.05, 0.1, 0.15]
num_epochs = 1000

# Run validation (optional - comment out if you just want to train)
# train_errors, test_errors = nn_validation()

# Train final model
print("\nTraining final NLP model...")
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.01
num_epochs = 1000

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % (num_epochs / 10) == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'final loss = {loss.item():.4f}')

# Save NLP model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "models/data.pth"
torch.save(data, FILE)
print(f"NLP model saved to {FILE}")

# Test symptom prediction
print("\nTesting symptom prediction...")
sentence = "My head hurts"
sentence = nltk.word_tokenize(sentence)
X = bag_of_words(sentence, all_words)
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X)

output = model(X)
_, predicted = torch.max(output, dim=1)
tag = tags[predicted.item()]

probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]

print(f"Input: 'My head hurts'")
print(f"Predicted symptom: {tag}")
print(f"Probability: {prob.item():.4f}")

# ============================================================================
# PART 2: Disease Prediction Model Training
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: Disease Prediction Model Training")
print("=" * 80)

# Load and transform disease data
df = pd.read_csv("data/dataset.csv")
df = df.drop_duplicates()

# Get unique symptoms
symptoms = np.concatenate((
    df.Symptom_1.unique(), df.Symptom_2.unique(), df.Symptom_3.unique(),
    df.Symptom_4.unique(), df.Symptom_5.unique(), df.Symptom_6.unique(),
    df.Symptom_7.unique(), df.Symptom_8.unique(), df.Symptom_9.unique(),
    df.Symptom_10.unique(), df.Symptom_11.unique(), df.Symptom_12.unique(),
    df.Symptom_13.unique(), df.Symptom_14.unique(), df.Symptom_15.unique(),
    df.Symptom_16.unique(), df.Symptom_17.unique()
))

symptoms_unique = list(set(symptoms))

# Create columns for each symptom
i = 18
for each in symptoms_unique:
    df.insert(i, each, 0)
    i = i + 1

df = df.fillna(0)

# Set symptom flags
for index, row in df.iterrows():
    disease_symptoms = [symptom for symptom in list(row)[1:] if symptom != 0]
    for each in disease_symptoms:
        df.at[index, each] = 1

# Drop old symptom columns
df = df.drop(columns=[
    'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
    'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
    'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
    'Symptom_16', 'Symptom_17'
])

df = df.loc[:, df.columns.notnull()]
df.columns = df.columns.str.replace(' ', '')
df = df.reindex(sorted(df.columns), axis=1)

# Load additional data
diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))

symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.map(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)

# Prepare training data
labels = df.to_numpy()[:, :1]
examples = df.to_numpy()[:, 1:]
list_of_symptoms = list(df.columns)[1:]

# Save list of symptoms
with open('data/list_of_symptoms.pickle', 'wb') as data_file:
    pickle.dump(list_of_symptoms, data_file)

print(f"Number of diseases: {len(labels)}")
print(f"Number of examples: {len(examples)}")
print(f"Number of symptoms: {len(list_of_symptoms)}")

# Cross validation function
def cross_validation(X_train, y_train, X_test, y_test, model_name, parameter_range=50):
    train_errors = []
    test_errors = []

    parameters = np.linspace(1, parameter_range, parameter_range, dtype=int)

    for parameter in parameters:
        if model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=parameter, metric='cosine')
        elif model_name == 'logreg':
            model = LogisticRegression(solver='liblinear', C=1/(parameter*20))
        elif model_name == 'dctree':
            model = DecisionTreeClassifier(splitter='random', max_depth=parameter)
        elif model_name == 'svm':
            model = SVC(C=1/(parameter*10))

        model.fit(X_train, y_train)
        learning_error = 1 - model.score(X_train, y_train)
        testing_error = 1 - model.score(X_test, y_test)
        train_errors.append(learning_error)
        test_errors.append(testing_error)

    if model_name == 'logreg':
        best_parameter_value = 1/(parameters[np.argmin(test_errors)]*20)
    elif model_name == 'svm':
        best_parameter_value = 1/(parameters[np.argmin(test_errors)]*10)
    else:
        best_parameter_value = parameters[np.argmin(test_errors)]

    return parameters, best_parameter_value, train_errors, test_errors

# Split data
X_train, X_test, y_train, y_test = train_test_split(examples, labels.ravel(), test_size=0.2)

# Train kNN
print("\nTraining kNN model...")
knn = KNeighborsClassifier(n_neighbors=6, metric='cosine')
knn.fit(X_train, y_train)
print(f"kNN accuracy: {knn.score(X_test, y_test):.4f}")

# Train Decision Tree
print("\nTraining Decision Tree model...")
clf = DecisionTreeClassifier(splitter='random', max_depth=100)
dc_tree = clf.fit(X_train, y_train)
print(f"Decision Tree accuracy: {dc_tree.score(X_test, y_test):.4f}")

# Train Logistic Regression
print("\nTraining Logistic Regression model...")
logreg = LogisticRegression(solver='liblinear', C=0.05)
logreg.fit(X_train, y_train)
print(f"Logistic Regression accuracy: {logreg.score(X_test, y_test):.4f}")

# Train SVM
print("\nTraining SVM model...")
svm = SVC(C=0.3)
svm.fit(X_train, y_train)
print(f"SVM accuracy: {svm.score(X_test, y_test):.4f}")

# Ensemble model
print("\nTraining Ensemble model...")
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def get_stacking():
    level0 = list()
    level0.append(('lr', LogisticRegression(solver='liblinear', C=0.03)))
    level0.append(('knn', KNeighborsClassifier(n_neighbors=6, metric='cosine')))
    level0.append(('dctree', DecisionTreeClassifier(splitter='random', max_depth=34)))
    level0.append(('svm', SVC(C=0.1)))

    level1 = LogisticRegression()
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return level0, model

models, model = get_stacking()

# Train and save ensemble model
model.fit(examples, labels.ravel())

with open('models/fitted_model.pickle2', 'wb') as modelFile:
    pickle.dump(model, modelFile)

print("Ensemble model saved to models/fitted_model.pickle2")

# Test prediction
print("\nTesting disease prediction...")
symptoms = ['stomach_pain', 'headache']
x_test = []

with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

for each in symptoms_list:
    if each in symptoms:
        x_test.append(1)
    else:
        x_test.append(0)

x_test = np.asarray(x_test)

with open('models/fitted_model.pickle2', 'rb') as modelFile:
    model_final = pickle.load(modelFile)

predicted = model_final.predict(x_test.reshape(1, -1))[0]
print(f"Input symptoms: {symptoms}")
print(f"Predicted disease: {predicted}")

description = diseases_description.loc[diseases_description['Disease'] == predicted.strip(" ").lower(), 'Description'].iloc[0]
print(f"Description: {description}")

print("\n" + "=" * 80)
print("Training completed successfully!")
print("=" * 80)
