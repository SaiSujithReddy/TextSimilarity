# import functions 
import numpy as np
import pandas as pd
import pickle
import gensim
from scipy import spatial
import heapq

# Returns top five elemets 
def find_top_five_max_element_list(given_list):
    list_backup = []
    list_backup = given_list
    values = heapq.nlargest(5,list_backup) # Get top five largest values
    indexes = np.argsort(given_list)[-5:] # Get indexes of top five largest values in reverse order
    top_five_elements = [indexes,values]
    print(top_five_elements)
    return top_five_elements

#Loading saved variables from a saved pkl file
with open ('Highlights product features.pkl','rb') as f:
    X_cosine_matrix_load = pickle.load(f) # Cosine matrix for given sentence, Not needed for now
    X_vector_space_load = pickle.load(f) # Vector space, for probing it is (14436, 300)
    content_sent_load = pickle.load(f) # List of sentences preserving order
    df_agent_unique_load = pickle.load(f) # Complete dataframe

# Get ids from original dataframe
def lookup_dataframe(sentence,cosine_similarity_number):
    df = pd.DataFrame()
    df = df_agent_unique_load.loc[df_agent_unique_load['content'] == sentence]
    row_id = df.iloc[0]['id']
    company_id = df.iloc[0]['company_id']
    relevant_data = [row_id,company_id,cosine_similarity_number]
    return relevant_data

# Transform sentence to vector
def get_vector_representation(sentence):
    content_list = []
    content_list.append(sentence.split(' '))
    
    vector_context = []
    word_vectors = model.wv
    vector_sentence = []
    for x in range(len(content_list[0])):
        if content_list[0][x] in word_vectors.vocab:
            vector_sentence.append(model[content_list[0][x]])
    if len(vector_sentence)!=0:
        vector_context.append(np.mean(np.array(vector_sentence), axis=0))
    else:
        vector_context.append(np.zeros(300))
    return vector_context

def calculate_cosine_similarity(x,y):
    return (1 - spatial.distance.cosine(x, y))

def calculate_sentence_similarity(sentence_vector):
    X_cosine_matrix = []
    for j in range(0,len(content_sent_load)):
        X_cosine_matrix.append(calculate_cosine_similarity(sentence_vector,X_vector_space_load[j]))
    return X_cosine_matrix

def display_items_reversed_order(items):
    content_data = []
    for i, j in zip(reversed(items[0]),items[1]):
        content_data.append('''<b> Conversation: </b> ''' +str(content_sent_load[i][0]))
        scores=lookup_dataframe(content_sent_load[i][0],j)
        content_data.append('''<b> Company ID: </b>''' + str(scores[1])+'''<b> ; Agent ID: </b>''' + str(scores[0])+'''<b> ; Cosine Similarity: </b>''' + str(round(scores[2],3)))
    return content_data
       

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save('google_vector_model')
print("Google vector corpus model saved.")
