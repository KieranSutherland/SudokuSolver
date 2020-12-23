from keras.datasets import mnist
import tensorflow as tf
import numpy as np

def main(): 
    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, test_data = training_data / 255, test_data / 255
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(training_data, training_labels, epochs=50) # loop through test data 50 times

    model.evaluate(test_data, test_labels)

    predictions = model.predict(test_data)
    np.set_printoptions(suppress=True)

    print(model.summary())

    print("Test number label: " + str(test_labels[0]))
    print("Actual model prediction result: " + str(predictions[0]))

    model.save("models/digits_model")

if __name__ == "__main__":
    main()