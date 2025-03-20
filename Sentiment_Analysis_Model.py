import tensorflow
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

max_features=10000
max_length=100
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train= sequence.pad_sequences(x_train,maxlen=max_length)
x_test= sequence.pad_sequences(x_test,maxlen=max_length)
embedding_dim=100
hidden_units=64
#Sequential model code
model=Sequential()
model.add(Embedding(input_dim=max_features,output_dim=embedding_dim,input_length=max_length))
model.add(LSTM(hidden_units))
model.add(Dense(1,activation='sigmoid'))
#Compiling and Training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))

def predict_sentiment(model,text):
  word_to_index=imdb.get_word_index()
  words=text.lower().split()
  f_tokens=[word_to_index[word]+ 3 for word in words if word in word_to_index]
  processed= sequence.pad_sequences([f_tokens],maxlen=max_length)
  prediction=model.predict(processed)[0][0]
  if (prediction > 0.5):
      print("Chatbot : I can see that the Sentiment here is positive !")
  elif (prediction < 0.5):
      print("Chatbot : This sentiment seems to be a negative one !")
  else:
      print("Chatbot : Hmm.. This one seems kind of neutral...")

print("\n\n\t\t\t\t    Welcome to the SAC - The Sentiment Analysis Chat Bot\n     ")
print("You can Chat with the chat bot and it will give you your sentiment as per its training. If you want the chatbot to stop type 'stop'.\n\n")
while True:
  text=input("\nYou : ")
  if text == 'stop' or text == 'STOP':
    print("\nChatbot : Thank you for talking with me. Have a great day !")
    break
  else:
    predict_sentiment(model,text)
    continue

