# Text Similarity

## Objective
<p> This project finds similar sentences for an agent's conversation. The dataset used for this purpose is chat conversations of multiple companies' customer support. By finding similar conversations, we intend to give better guidance and support to the agent, thereby improving customer's experience. This tool can be used to both help/support the agent as well as measure their performance.</p>

## Data
<p> Below is a snapshot of the chat conversation data: </p>

```| -------------------Content ------------------------------| ------Label--------- |
| How are you doing ?                                      | Greeting |
| Please find product details at this location             | Product features |
| You could save potentially $100/year with our product    | Benefits to customer |
| Can i get refund ?                                       | Refund |
| This program includes 1 online course every semester     | program features to customer | 
| I am looking for any discount                            | Discount |
| Can i get the number on the card for purchase            | Close attempt |
| May I please know why are you dissatified with product ? | Enquires for pain points |
| Sorry for the inconvenience that has been caused to you  | Pleases customer |
| This program is much better than our competitor          | Upsells the product |

```

## Methodology
<p> Word 2 Vector representation of all the agents' conversations is created using Google Word2Vec model. When a new sentence is given, a word2vec representation is computed. Cosine similarity is calculated between the sentence and all the vectors. By finding similar sentences/conversations, agent can find the right responses from the repository. A flask based web application gives a web interface to find the similar sentences. </p>

## Pipeline

<p align="center">
<img src="https://github.com/SaiSujithReddy/TextSimilarity/blob/master/Screen%20Shot%202017-10-08%20at%2010.17.59%20PM.png" alt="Sentence similarity Pipeline" width="600px">
</p>

## Demo
<p> Video of the Web application demo is [here] (https://youtu.be/Ity2xKESYPo)</p>

## Requirements
```
gensim==2.3.0
ipython==5.4.1
Keras==2.0.5
matplotlib==2.0.2
nltk==3.2.5
numpy==1.13.0
pandas==0.20.2
scikit-image==0.13.0
scikit-learn==0.18.2
scipy==0.19.1
tensorflow==1.2.1
```


### helper_functions_v1:

<p> Includes model, helper functions.
Inputs: a pkl file with skill_name
No output expected. </p>

### sentence_similarity.py:

<p> Includes python functions that calls helper functions and returns output.
Inputs: Input sentence (for which you need to find similar sentences)
Output: Five most similar sentences where each sentence has agent id/ company id / cosine similarity measure </p>


