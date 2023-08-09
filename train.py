from keras.models import Model
from keras.layers import Embedding, LSTM, Activation, BatchNormalization, Dense, Bidirectional
from readData import x_train, y_train, tokenizer, x_test, y_test
from keras.callbacks import EarlyStopping

class LSTMModel(Model):
    def __init__(self, input_shape):
        super(LSTMModel, self).__init__()
        self.embedding = Embedding(input_dim=input_shape, output_dim=100)
        self.lstm = Bidirectional(LSTM(128, activation='tanh', use_bias=False, dropout=0.2, recurrent_dropout=0.2))
        self.dense1 = Dense(200, activation='relu')
        self.batchNorm = BatchNormalization()
        self.dense2 = Dense(100, activation='relu')
        self.dense3 = Dense(50, activation='relu')
        self.outputDense = Dense(1, activation='sigmoid')
        self.activation = Activation('sigmoid')
    def call(self, x):
        x  = self.embedding(x)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.batchNorm(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.outputDense(x)
        x = self.activation(x)
        return x
    
earlyStawwwpppp = EarlyStopping(restore_best_weights=True, patience=3)
    
input_shape = len(tokenizer.word_index) + 1

EPOCHS = 30

model = LSTMModel(input_shape=input_shape)

model.build(input_shape=(input_shape, None))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=earlyStawwwpppp)

model.save("F:/DisasterTweets/model") 

    