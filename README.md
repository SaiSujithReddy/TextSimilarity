# Text Similarity

## Objective
<p> This project classifies conversational phrases into different categories based on the content. The network is trained on customer support conversations. Only agent's conversation is used for this purpose. The different categories of the conversation indicate the information the agent is providing/gathering. By predicting the classification category,the agent can be provided by more targeted documentation to help the customer. This tool can be used to both help/support the agent as well as measure their performance.</p>

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


