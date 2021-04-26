#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


with open('train_qa.txt','rb') as f:
    training_data = pickle.load(f)


# In[3]:


training_data


# In[4]:


with open('test_qa.txt','rb') as f:
    test_data = pickle.load(f)


# In[5]:


test_data


# In[6]:


t = (len(training_data),len(test_data))


# In[7]:


my_list = list(t)


# In[8]:


my_list


# In[9]:


type(training_data)


# In[10]:


type(training_data[0])


# In[11]:


training_data[0]


# In[12]:


training_data[0][0]


# In[13]:


' '.join(training_data[0][0])


# In[14]:


' '.join(training_data[0][1])


# In[15]:


all_data = test_data + training_data


# In[16]:


len(all_data)


# In[17]:


set(training_data[0][1])


# In[ ]:





# In[18]:


vocab = set()

for story,question,answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
    vocab.add('yes')
    vocab.add('no')
    


# In[19]:


vocab_len = len(vocab) + 1


# In[20]:


all_data[0][0]


# In[21]:


all_story_len = [len(data[0]) for data in all_data]
    


# In[22]:


print(all_story_len)


# In[23]:


max_story_len = max(all_story_len)


# In[24]:


max_story_len


# In[25]:


all_question_len = [len(data[1]) for data in all_data]
max_question_len = max(all_question_len)


# In[26]:


max_question_len


# In[ ]:





# In[ ]:





# In[27]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[28]:


tokenizer = Tokenizer(filters = [])


# In[29]:


tokenizer.fit_on_texts(vocab)


# In[30]:


tokenizer.word_index


# In[31]:


train_story_texts = []
train_question_texts = []
train_answers = []


# In[32]:


for story,question,answer in training_data:
    train_story_texts.append(story)
    train_question_texts.append(question)
    train_answers.append(answer)
    
    


# In[33]:


train_story_texts


# In[34]:


train_question_texts


# In[35]:


train_answers


# In[36]:


train_story_sequence = tokenizer.texts_to_sequences(train_story_texts)


# In[37]:


train_story_sequence


# In[38]:


type(train_story_texts)


# In[39]:


def vectorize_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_question_len=max_question_len):
    
    
    X = []
    
    Xq = []
    
    y = []
    
    for story,question,answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        
        
        Y = np.zeros(len(word_index)+1)
        
        Y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        y.append(Y)
        
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_question_len),np.array(y))    
        
        
        
        
        
        
        
        


# In[40]:


# vectorizing stories,questions and answers by calling vectorize_space function now


# In[41]:


inputs_data,questions_data,answers_data = vectorize_stories(training_data)


# In[42]:


inputs_test,questions_test,answers_test = vectorize_stories(test_data)


# In[43]:


inputs_data


# In[44]:


tokenizer.word_index['yes']


# In[45]:


tokenizer.word_index['no']


# In[46]:


from keras.models import Sequential,Model


# In[47]:


from keras.layers.embeddings import Embedding


# In[48]:


from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM


# In[49]:



input_sequence = Input((max_story_len, ))
question = Input((max_question_len, ))


# In[50]:


input_sequence


# In[ ]:





# In[ ]:





# In[51]:


vocab_size = len(vocab) + 1


# In[52]:


# input encoder M and C


# In[53]:


input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))


# In[54]:


input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[55]:


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))


# In[56]:


input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# In[57]:


match = dot([input_encoded_m,question_encoded],axes=(2,2))


# In[58]:


match = Activation('softmax')(match)


# In[59]:


response = add([match,input_encoded_c])
response = Permute((2,1))(response)


# In[60]:


answer = concatenate([response,question_encoded])


# In[61]:


answer


# In[62]:


answer = LSTM(32)(answer)


# In[63]:


answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)


# In[64]:


answer = Activation('softmax')(answer)


# In[65]:


model = Model([input_sequence,question],answer)


# In[66]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[67]:


model.summary()


# In[68]:


history = model.fit([inputs_data,questions_data],answers_data,batch_size=32,epochs=5,validation_data=([inputs_test,questions_test],answers_test))


# In[69]:


history.history.keys()


# In[70]:


plt.plot(history.history['accuracy'])


# In[71]:


plt.plot(history.history['loss'])


# In[72]:


preds = model.predict(([inputs_test,questions_test]))


# In[73]:


type(preds)


# In[74]:


preds.argmax()


# In[75]:


preds[0]


# In[76]:


np.argmax(preds[0])


# In[82]:


val_max = np.argmax(preds[0])


# In[83]:


val_max


# In[84]:


for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key


# In[85]:


k


# In[86]:


preds[0][val_max]


# In[87]:


vocab


# In[ ]:




