import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import string as str

words=[]
classes = []
docs_x = []
docs_y = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        docs_x.append(w)
        docs_y.append(intent["tag"])
        # add to our classes list
        
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
output = []

out_empty = [0 for _ in range(len(classes))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)


    output_row = out_empty[:]
    output_row[classes.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output) 


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains 32 neurons
# equal to number of intents to predict output intent with softmax
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=[174],activation ='relu'),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32), 
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(len(output[0]), activation="softmax"),
])

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(training, output, epochs=4000, batch_size=8)
model.save('chatbot_model.h5')

print("model created")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

# def chat():
#     print("Start talking with the bot (type quit to stop)!")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == "quit":
#             break

#         results = model.predict(np.array([bag_of_words(inp, words)]))[0]
#         results = np.around(results, decimals=5)
#         print(results)# results_index = np.argmax(results)
        # tag = classes[results_index]
        
        # if results[results_index] > 0.5:
        #     for tg in intents["intents"]:
        #         if tg['tag'] == tag:
        #             responses = tg['responses']

        #     print(random.choice(responses))
        # else:
        #     print("I didn't quite get that, try again or ask another question.")
#chat()
