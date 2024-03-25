import tensorflow as tf
import numpy as np

NumberData = []
TestData =[]

for x in range(0, 50000):
    number = np.random.randint(low = 0, high = 9)
    NumberData.append(float(number))

for x in range(0, 500):
    number = np.random.randint(low = 0, high = 9)
    TestData.append(float(number))

NumberLabels = NumberData

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (1,)),
    tf.keras.layers.Dense(35,activation='relu'),
    tf.keras.layers.Dense(35,activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(
    NumberData,
    NumberLabels,
    epochs=6,
)

prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print("What is your number?\n")
NumberToGuess = input()
NumberToGuess = float(NumberToGuess)
NumberToGuess = np.array(NumberToGuess)
NumberToGuess = NumberToGuess.reshape(1, )
prediction = prob_model.predict(NumberToGuess)
predicted_number = np.argmax(prediction)
print("Your number is probably: " + str(predicted_number))