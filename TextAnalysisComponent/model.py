#import tensorflow as tf
#from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, BertTokenizer, TFBertForSequenceClassification

#from transformers import AutoTokenizer, AutoModelForSequenceClassification

#Load the tokenizer
# tokenizer_roberta = AutoTokenizer.from_pretrained("izyaan/redditmi-roberta-base")
# tokenizer_bert = AutoTokenizer.from_pretrained("izyaan/redditmi-bert-base-uncased")

#Load the model
# model_roberta = AutoModelForSequenceClassification.from_pretrained("izyaan/redditmi-roberta-base", from_tf=True)
# model_bert = AutoModelForSequenceClassification.from_pretrained("izyaan/redditmi-bert-base-uncased", from_tf=True)

# #Define the optimizer, loss, and metric
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Accmetric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

# #Compile the model
# model_roberta.compile(optimizer=optimizer, loss=loss, metrics=[Accmetric])
# model_bert.compile(optimizer=optimizer, loss=loss, metrics=[Accmetric])

# #Load the saved weights
# model_roberta.load_weights('modelweights/RoBERTa_model_weights.h5')
# model_bert.load_weights('modelweights/BERTmodel_weights.h5')

