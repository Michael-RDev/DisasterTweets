from readData import tokenizer, max_length, tweet_type
import tensorflow as tf
from keras.utils import pad_sequences
import pandas as pd 

loaded_mod = tf.keras.models.load_model('F:/DisasterTweets/model')

tweet_type = list(set(tweet_type))

def predictTweets(input_txt:str):
    tokenizer_input = tokenizer.texts_to_sequences([input_txt]) 
    padded_words = pad_sequences(tokenizer_input, maxlen=max_length)
    prediction = loaded_mod.predict(padded_words)[0]
    thres = 0.5
    prediction_num = [tweet_type[1] if prediction > thres else tweet_type[0]]
    prediction_final = ["disaster" if prediction_num == [1]  else "non disaster"]
    return prediction_final



if __name__ == "__main__":
    writeFile = open('test_results.txt','r+', encoding='utf-8')
    testFile = pd.read_csv('F:/DisasterTweets/data/test.csv')
    test_text = testFile['text'].values
    
    for text in test_text:
        prediction = predictTweets(text)
        writeFile.write(f"Text: {text} \n")
        writeFile.write(f"Prediction: {prediction} \n")
        writeFile.write("=" * 50 + "\n")
    writeFile.close()