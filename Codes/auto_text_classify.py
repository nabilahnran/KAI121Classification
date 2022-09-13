import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

pd.set_option('display.max_columns', 10)

testing_portion = 0.2
oov_tok = '<OOV_TOK>'
vocab_size = 10000
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
max_length = 150

#load data files
data_vocab = pd.read_csv('KAI121mentions-merged.csv')# *
# * changeable. if you want to use file with no stopwords, change the file name to
# KAI121mentions - removed_stpwrds_number.csv
predict_file = input('Masukkan path file data yang ingin diklasifikasikan\nJika pada folder yang '
                    'sama masukkan nama filenya saja (format .csv) \n Masukkan disini: ')
result_file = input ('Masukkan nama file baru untuk menyimpan hasil klasifikasi (format .csv)\n '
                     'Masukkan disini: ')
data_predict = pd.read_csv(predict_file, delimiter=';')
data_vocab.drop(columns=['Unnamed: 0'], inplace = True)
data_predict.drop(columns=['Unnamed: 0'], inplace = True)

#uncomment this two lines below if you are using no stopwords data
"""data_vocab.text=data_vocab.text.astype(str)
data_predict.text=data_predict.text.astype(str)"""

#drop unused columns for classification
data_vocab.drop(columns=['tanggal'],
                inplace = True)

#load saved model
sent_model = load_model('sent_nosplit_stpwrds.h5') # **
topic_model = load_model('topic_nosplit_stpwrds.h5') # **
# ** changeable. if you want to use file with no stopwords, change the name file to
# sent_model = sent_nosplit_nostpwrds.h5, topic_model = topic_nosplit_nostpwrds.h5

#cleaning data function
def cleaning_data(data_predict):
    #Remove punctuation and emojis
    data_predict['text'] = data_predict['text'].str.replace('[^\w\s]', '')
    # Lowering Case
    data_predict['text'] = data_predict['text'].str.lower()
    # remove URLs
    data_predict['text'] = data_predict['text'].replace(r'http\S+', '', regex=True)\
        .replace(r'www\S+', '', regex=True)
    # remove newlines
    data_predict['text'] = data_predict['text'].str.replace('\n', ' ')
    # replace two space to one
    data_predict['text'] = data_predict['text'].str.replace('\s\s+', ' ', regex=True)
    # remove leading space
    data_predict['text'] = data_predict['text'].replace('^ +| +$', '', regex=True)
    #remove number
    data_predict['text'] = data_predict['text'].str.replace('\d+', '')
    #remove single letter
    data_predict['text'] = data_predict['text'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
    # manipulate NaN values
    data_predict.loc[(data_predict['text'].isnull()), 'text'] = 'nulltext'
    #remove stopwords
    #uncomment this section if you use no stopwords input data
    """f = open("id-stopwords.txt", "r")
    list = f.read().splitlines()
    print(list)
    data_predict['text'] = data_predict['text'].apply(lambda x: ' '.join([word for word in x.split()
                                                                        if word not in (list)]))"""

    return data_predict

#preprocessed data function
def preprocessed(data_vocab):
    data_set = data_vocab
    text_set = data_set['text']
    sent_set = data_set['sentiment']
    topic_data = data_set['topic']

    #tokenization
    tokenizer = Tokenizer(vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(text_set)
    text_word_index = tokenizer.word_index

    sent_tokenizer = Tokenizer()
    sent_tokenizer.fit_on_texts(sent_set)
    sent_word_index = sent_tokenizer.word_index

    topic_tokenizer = Tokenizer()
    topic_tokenizer.fit_on_texts(topic_data)
    topic_word_index = topic_tokenizer.word_index

    print("setences dict:")
    print(text_word_index)
    print("sentiment dict:")
    print(sent_word_index)
    print("topic dict:")
    print(topic_word_index)

    return tokenizer, text_word_index, sent_tokenizer, sent_word_index,\
           topic_tokenizer, topic_word_index

#predict sentiment label function
def predict_sentiment(data_predict):
    input = np.array(data_predict['text'])

    #predict with the model
    prediction = sent_model.predict(np.array(pad_sequences(tokenizer.texts_to_sequences(input),
                                                           padding=padding_type, maxlen=max_length,
                                                           truncating=trunc_type)))
    list_result = []
    for row in range(len(prediction)):
        result = prediction[row].tolist().index(np.max(prediction[row]))
        list_result.append(result)
        print("prediction- " + str(row) + " : " + str(result) + " with " + str(
            100 * max(prediction[row].tolist() * 100)) + " percentage")
    pre_result = []
    for s in list_result:
        pre_result.append(sent_tokenizer.index_word[s])

    #place the prediction to the new column
    data_predict['Sentiment Prediction'] = np.array(pre_result)
    print(data_predict[['text', 'Sentiment Prediction']])
    return data_predict

#predict topic label function
def predict_topic(data_predict):
    input_text = np.array(data_predict['text'])
    #prepare the prediction input (text+predicted sentiment)
    input_text = pad_sequences(tokenizer.texts_to_sequences(input_text),
                                                           padding=padding_type, maxlen=max_length,
                                                           truncating=trunc_type)
    input_sent = np.array(data_predict['Sentiment Prediction'])
    input_sent = pad_sequences(sent_tokenizer.texts_to_sequences(input_sent), maxlen=1)
    input = np.concatenate([input_text, input_sent], axis=1)

    #predict with the model
    prediction = topic_model.predict(np.array(input))
    list_result = []
    for row in range(len(prediction)):
        result = prediction[row].tolist().index(np.max(prediction[row]))
        list_result.append(result)
        print("prediction- " + str(row) + " : " + str(result) + " with " + str(
            100 * max(prediction[row].tolist() * 100)) + " percentage")
    pre_result = []
    for s in list_result:
        pre_result.append(topic_tokenizer.index_word[s])

    #place the prediction to a new column
    print("\n\n============================= FINAL RESULT BELOW =============================")
    data_predict['Topic Prediction'] = np.array(pre_result)
    print(data_predict[['text', 'Sentiment Prediction', 'Topic Prediction']])
    print("\n*** Hasil klasifikasi berhasil disimpan ke dalam file bernama " + result_file +" ***")
    return data_predict

if __name__ == '__main__':
    print(data_predict.columns)
    print(data_vocab.columns)
    #clean data first
    data_predict = cleaning_data(data_predict)
    #pre processing data
    tokenizer, text_word_index, sent_tokenizer, sent_word_index, topic_tokenizer, topic_word_index\
        = preprocessed(data_vocab)
    #predict sentiment label
    data_predict = predict_sentiment(data_predict)
    #predict topic label
    data_predict = predict_topic(data_predict)
    #save the result file to a new csv file
    data_predict.to_csv(result_file)