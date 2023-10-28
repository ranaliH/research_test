import emoji
import torch
from collections import Counter
from TextAnalysisComponent.reddit import retrieve_user_data
import tensorflow as tf
from transformers import pipeline

def preprocess_emojis(texts):
    processed_texts = []
    
    for text in texts:
        processed_text = emoji.demojize(text)
        
        processed_text = processed_text.replace(":", "").replace("_", " ")
        
        processed_texts.append(processed_text)
    
    return processed_texts


def predict_classes(texts, threshold):
    label_mapping_reverse = {
        'LABEL_0': 'depression',
        'LABEL_1': 'Anxiety',
        'LABEL_2': 'bipolar',
        'LABEL_3': 'BPD',
        'LABEL_4': 'schizophrenia',
        'LABEL_5': 'autism'
    }
    predicted_classes = []

    # Preprocess the texts
    processed_texts = preprocess_emojis(texts)
    max_length = 512
    # Create a text classification pipeline for BERT
    classifier_bert = pipeline("text-classification", model="izyaan/redditmi-bert-base-uncased", tokenizer="izyaan/redditmi-bert-base-uncased")

    # Create a text classification pipeline for RoBERTa
    classifier_roberta = pipeline("text-classification", model="izyaan/redditmi-roberta-base", tokenizer="izyaan/redditmi-roberta-base")

    for text in processed_texts:
        # Truncate the input text if it's too long
        truncated_text = text[:max_length]
        # Make predictions using the BERT pipeline
        results_bert = classifier_bert(truncated_text)

        # Make predictions using the RoBERTa pipeline
        results_roberta = classifier_roberta(truncated_text)

        # Extract predicted labels and confidence scores
        predicted_label_bert = results_bert[0]['label']
        predicted_class_confidence_bert = results_bert[0]['score']

        predicted_label_roberta = results_roberta[0]['label']
        predicted_class_confidence_roberta = results_roberta[0]['score']

        # Calculate average ensemble confidence score
        avg_confidence = (predicted_class_confidence_bert + predicted_class_confidence_roberta) / 2
        # print(avg_confidence)
        if avg_confidence > threshold:
            predicted_class = label_mapping_reverse[predicted_label_bert]
        else:
            predicted_class = 'None'

        predicted_classes.append(predicted_class)
        
    return predicted_classes



def get_most_predicted_class(username):
    # Assuming `df1` contains the retrieved user data
    df1 = retrieve_user_data(username)
    if df1.empty:
        print('no relevant posts')
        return 'None'
    #print(type(df1))
    #df = pd.DataFrame(df1)
    # Assuming the post/comment content is stored in the 'content' column
    # print(df1['content'])
    texts = df1['content'].tolist()

    # print('predict func')
    # Call the `predict_classes` function with the `texts` and desired threshold
    results = predict_classes(texts, threshold=0.95)  # Adjust the threshold as needed
    # print('results ', results)
    # Count the occurrences of each predicted class
    class_counts = Counter(results)
    # print('class count ', class_counts)
    # Find the most common predicted class
    most_predicted_class = class_counts.most_common(1)[0][0]

    return most_predicted_class
    #return class_counts

