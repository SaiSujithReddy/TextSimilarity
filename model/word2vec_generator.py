import pandas as pd
import gensim
import numpy as np
import pickle
import random
from scipy import spatial
from numpy import array

skill_name_unique_list = ["Attempts to close the sale","Fact-gathering question","Highlights benefits to customer",
                          "Attempts to upsell","Attempts to schedule a next step","Test close"]
df = pd.read_csv('data/four_companies_event_level_data.csv',sep='|', quotechar='"')

def get_processed_data(df,skill_name):
    df_agent = df.loc[(df.is_customer == 'f') & (df.skill_name == skill_name_value)]
    df_agent.skill_name.fillna(value='none',inplace=True)
    df_agent_unique = df_agent[df_agent.content.str.contains("airconditioning,") == False]
    df_agent_unique = df_agent_unique.drop_duplicates('content')    
    df_agent_unique_min_length = df_agent_unique[~(df_agent_unique.content.str.len() < 50)]

    df_agent_unique_min_length = df_agent_unique_min_length[~(df_agent_unique.content.str.len() > 1000)]
    
    df_agent_unique_min_length = df_agent_unique_min_length[(df_agent_unique['content'].notnull())]
    print("df_agent_unique_min_length shape is ", df_agent_unique_min_length.shape)
    return df_agent_unique_min_length

def create_content(df_agent_unique_min_length):
    content_sent = []
    for index, row in df_agent_unique_min_length.iterrows():
        content_sent.append([row['content']])
    print(len(content_sent))
    
    content_list = []
    for i in range(len(content_sent)):
        content_list.append(content_sent[i][0].split(' '))
    print(len(content_list))
    return content_list

def generate_vector(content_list):
    for i in range(len(content_list)):
        vector_sentence = []
        for x in content_list[i]:
            if x in word_vectors.vocab:
                vector_sentence.append(model[x])
        if len(vector_sentence)!=0:
            mean_value = np.mean(np.array(vector_sentence), axis=0)
            #print("value of mean_value", mean_value)
            if(mean_value.all != 0):
                vector_context.append(np.mean(np.array(vector_sentence), axis=0))
            else:
                print("Mean value of vector context value is zero, sentence number is ", i)
                print("sentence is ", content_list[i])
                vector_context.append(np.random.rand(300))        
        else:
            print("vector context value is zero, sentence number is ", i)
            print("sentence is ", content_list[i])
            vector_context.append(np.random.rand(300))
    return vector_context

def create_pickle():
    for skill_name_value in skill_name_unique_list:
        df_agent_unique_min_length=get_processed_data(df,skill_name)
        content_list=create_content(df_agent_unique_min_length)
        X = array(generate_vector(content_list)
        for i in range(0,len(X)):
            if np.all(X[i]==0):
                X[i] = np.random.rand(300)
        X_truncated = X[0:10]
        X_cosine_matrix = []
        for i in range(len(X_truncated)):
            X_cosine_matrix.append([i]*len(X))
            for j in range(0,len(X)):
                X_cosine_matrix[i][j] = calculate_cosine_similarity(X_truncated[i],X[j])

        #Saving variables in to a file

        with open(skill_name_value+'.pkl','wb') as f:
            pickle.dump(X_cosine_matrix,f)
            pickle.dump(X,f)
            pickle.dump(content_sent,f)
            pickle.dump(df_agent_unique_min_length,f)
            print("Saved ", skill_name_value+'.pkl', " file")
