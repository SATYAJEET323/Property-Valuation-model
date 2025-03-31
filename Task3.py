# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from google.colab import drive
import pandas as pd
import re
from simpletransformers.classification import ClassificationModel
import os
import shutil
import time
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
#1.  DataLoading -------------------------------------------------------------------------------------------------------------------------------
# Step 1: Mount Google Drive

# Step 2: Define the path to your dataset in Google Drive
file_path = '/Property_val_dataset.csv'  # Change this to the actual path

# Step 3: Load the dataset (assuming it's a CSV file)
df = pd.read_csv(file_path)
# Exploratory Data Analysis (EDA)
print(df.info())  # Overview of data structure
print(df['Prediction'].value_counts())  # Class distribution

# Split dataset into train and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
# Preparing the data in the correct format for SimpleTransformers
train_df = pd.DataFrame({
    'Prediction': train_data['Prediction'],
    'Prediction': train_data['Prediction']
})

val_df = pd.DataFrame({
    'Prediction': val_data['Prediction'],
    'Prediction': val_data['Prediction']
})
train_df['Prediction'] = train_df['Prediction'].astype(str)
val_df['Prediction'] = val_df['Prediction'].astype(str)

# Define the label mapping dictionary before using it
label_mapping = {
    'positive': 1,
    'negative': 0,
    'neutral': 2
}

# Convert text labels to numeric values
train_df = pd.DataFrame({
    'text': train_data['Input'],
    'labels': train_data['Prediction'].map(label_mapping)
})

val_df = pd.DataFrame({
    'text': val_data['Input'],
    'labels': val_data['Prediction'].map(label_mapping)
})

# 2 . Text Preprocessing -------------------------------------------------------------------------------------------------------------------

# Define a function to clean text data
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()
    return text

# Convert 'labels' column to string type and apply cleaning
train_df['labels'] = train_df['labels'].astype(str).apply(clean_text)
val_df['labels'] = val_df['labels'].astype(str).apply(clean_text)

print(train_df.head())  # Check output


# 3 . Text Embedding using BERT and RoBERTa --------------------------------------------------------------------------------------------------
# Create a BERT model for text classification
bert_model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, use_cuda=False)  # Set use_cuda=True if using a GPU

# Create a RoBERTa model for text classification
roberta_model = ClassificationModel('roberta', 'roberta-base', num_labels=2, use_cuda=False)  # Set use_cuda=True if using a GPU


# 4 . Model Training with BERT and RoBERTa:--------------------------------------------------------------------------------------------------
# Preparing the data in the correct format for SimpleTransformers
# Assuming 'Prediction' column contains the text and you want to predict the same 'Prediction'
# If you have a separate column for labels, replace 'Prediction' with that column name

# Create a dictionary to map unique string labels to numerical labels starting from 0
unique_labels = train_data['Prediction'].unique()
num_labels = len(unique_labels)  # Get the actual number of unique labels

label_mapping = {label: idx for idx, label in enumerate(unique_labels)}


train_df = pd.DataFrame({
    'text': train_data['Input'],  # Changed 'Prediction' to 'text' for the input text
    'labels': train_data['Prediction'].map(label_mapping)  # Map string labels to numerical labels
})

val_df = pd.DataFrame({
    'text': val_data['Input'],  # Changed 'Prediction' to 'text' for the input text
    'labels': val_data['Prediction'].map(label_mapping)  # Map string labels to numerical labels
})

# Update the model definition with the correct number of labels
bert_model = ClassificationModel('bert', 'bert-base-uncased', num_labels=num_labels, use_cuda=False)  # Update num_labels

# Now you can train your model
bert_model.train_model(train_df)

# !rm -rf outputs/


df = pd.read_csv(file_path)
# Exploratory Data Analysis (EDA)
print(df.info())
print(df['Prediction'].value_counts())

# Split dataset into train and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Create label mapping from the full dataset
unique_labels = df['Prediction'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Convert text and labels into DataFrame
train_df = pd.DataFrame({
    'text': train_data['Input'],
    'labels': train_data['Prediction'].map(label_mapping)
})

val_df = pd.DataFrame({
    'text': val_data['Input'],
    'labels': val_data['Prediction'].map(label_mapping)
})

# Ensure 'Prediction/' is clean before training
output_dir = "outputs/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Define the model correctly
roberta_model = ClassificationModel(
    model_type='roberta',
    model_name='roberta-base',
    num_labels=len(label_mapping),
    use_cuda=False
)

# Train the model
roberta_model.train_model(train_df, overwrite_output_dir=True)

# 5 . Evaluation on Validation Set:------------------------------------------------------------------------------------------------------------


df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
print(df.info())  # Overview of data structure
print(df['Prediction'].value_counts())  # Class distribution

# Split dataset into train and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)



# Preparing the data in the correct format for SimpleTransformers
# Assuming 'Output' column contains the text and you want to predict the same 'Output'
# If you have a separate column for labels, replace 'Output' with that column name

# Create a dictionary to map unique string labels to numerical labels starting from 0
unique_labels = train_data['Prediction'].unique()
num_labels = len(unique_labels)  # Get the actual number of unique labels

label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

train_df = pd.DataFrame({
    'text': train_data['Input'],  # Changed 'Output' to 'text' for the input text
    'labels': train_data['Prediction'].map(label_mapping)  # Map string labels to numerical labels
})

val_df = pd.DataFrame({
    'text': val_data['Input'],  # Changed 'Output' to 'text' for the input text
    'labels': val_data['Prediction'].map(label_mapping)  # Map string labels to numerical labels
})


# Set up model arguments with custom hyperparameters
model_args = ClassificationArgs(
    num_train_epochs=3,       # Start with 3 epochs
    train_batch_size=8,       # Use a batch size of 8
    eval_batch_size=8,        # Same for evaluation
    learning_rate=3e-5,       # Learning rate
    max_seq_length=128,       # Max sequence length
    weight_decay=0.01,        # Weight decay
    warmup_steps=0,           # Optional: adjust based on total steps
    logging_steps=50,         # Log training progress every 50 steps
    save_steps=200,           # Save the model every 200 steps
)

# Function to clear the outputs directory
def clear_outputs_directory():
  # Try to remove the directory using shutil.rmtree
  try:
    shutil.rmtree("outputs/")
  except OSError as e:
    print(f"Error removing outputs/ directory: {e}")
  # Wait for 2 seconds to allow any processes to release the directory
  time.sleep(2)
  # Create the directory if it doesn't exist
  os.makedirs("outputs/", exist_ok=True)


# Train the BERT model
clear_outputs_directory()
bert_model = ClassificationModel('bert', 'bert-base-uncased', num_labels=num_labels, args=model_args, use_cuda=False)
bert_model.train_model(train_df, overwrite_output_dir=True)

# Train the RoBERTa model
clear_outputs_directory()
roberta_model = ClassificationModel('roberta', 'roberta-base', num_labels=num_labels, args=model_args, use_cuda=False)
roberta_model.train_model(train_df, overwrite_output_dir=True)

# Evaluate BERT on validation data
result_bert, model_outputs_bert, wrong_predictions_bert = bert_model.eval_model(val_df)

print("BERT Evaluation Results:")
print(result_bert)

# Evaluate RoBERTa on validation data
result_roberta, model_outputs_roberta, wrong_predictions_roberta = roberta_model.eval_model(val_df)

print("RoBERTa Evaluation Results:")
print(result_roberta)

# 6. Saving the Best Model----------------------------------------------------------------------------------------------------------------------

bert_model.save_model('bert_best_model')
roberta_model.save_model('roberta_best_model')
# !ls -l ./bert_best_model

# 7. Prediction on Real-World Input:------------------------------------------------------------------------------------------------------------

# Define the number of labels (ensure this is correctly set according to your dataset)
num_labels = 2  # Change this based on your classification task

# Set up model arguments
model_args = ClassificationArgs(
    num_train_epochs=3,
    train_batch_size=8,
    eval_batch_size=8,
    learning_rate=3e-5,
    max_seq_length=128,
    weight_decay=0.01,
    warmup_steps=0,
    logging_steps=50,
    save_steps=200,
    overwrite_output_dir=True,  # Ensure previous files are replaced
    output_dir='./bert_best_model'  # Specify save path explicitly
)

# Example training data (replace with actual training data)
train_data = {
    'text': ['example sentence 1', 'example sentence 2'],
    'labels': [0, 1]  # Ensure labels are numerical
}
train_df = pd.DataFrame(train_data)

# Create and train the BERT model
bert_model = ClassificationModel('bert', 'bert-base-uncased', num_labels=num_labels, args=model_args, use_cuda=False)
bert_model.train_model(train_df)

# Now load the model using the same path
bert_model = ClassificationModel('bert', './bert_best_model', use_cuda=False)

# Continue with your predictions
real_world_text = ['Bangalore', '3935 sq. ft', 'Rooftop access', 'Old house', 'gym']
predictions_bert, _ = bert_model.predict(real_world_text)
print(f"BERT Predictions: {predictions_bert}")


# Example training data (replace with actual training data)
train_data = {
    'text': ['example sentence 1', 'example sentence 2'],
    'labels': [0, 1]  # Ensure labels are numerical and match correct column name
}
train_df = pd.DataFrame(train_data)

# Ensure num_labels is calculated correctly
unique_labels = train_df['labels'].unique()  # Updated from 'Output' to 'labels'
num_labels = len(unique_labels)  # Get the actual number of unique labels

# Set up model arguments
model_args = ClassificationArgs(
    num_train_epochs=3,
    train_batch_size=8,
    eval_batch_size=8,
    learning_rate=3e-5,
    max_seq_length=128,
    weight_decay=0.01,
    warmup_steps=0,
    logging_steps=50,
    save_steps=200,
    overwrite_output_dir=True,
    output_dir='./bert_best_model'
)

# Create and train the BERT model
bert_model = ClassificationModel(
    'bert',
    'bert-base-uncased',
    num_labels=num_labels,
    args=model_args,
    use_cuda=False
)  # ✅ Fixed missing parenthesis

bert_model.train_model(train_df)

# Now load the model using the same path
bert_model = ClassificationModel('bert', './bert_best_model', use_cuda=False)

# Continue with your predictions
real_world_text = ['Bangalore', '3935 sq. ft', 'Rooftop access', 'Old house', 'gym']
predictions_bert, _ = bert_model.predict(real_world_text)
print(f"BERT Predictions: {predictions_bert}")

# !ls -l ./roberta_best_model

df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
print(df.info())
print(df['Prediction'].value_counts())

# Split dataset into train and validation sets
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Prepare data for SimpleTransformers
unique_labels = train_data['Prediction'].unique()
num_labels = len(unique_labels)

label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

train_df = pd.DataFrame({
    'text': train_data['Input'],
    'labels': train_data['Prediction'].map(label_mapping)
})

val_df = pd.DataFrame({
    'text': val_data['Input'],
    'labels': val_data['Prediction'].map(label_mapping)
})

# ----------------------
# RoBERTa Model Training and Saving
# ----------------------

# Remove the output directory if it exists
output_dir = './roberta_best_model'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create and train the RoBERTa model
roberta_model = ClassificationModel('roberta', 'roberta-base', num_labels=num_labels, use_cuda=False)
roberta_model.train_model(train_df, output_dir=output_dir) #Removed overwrite_output_dir=True,



# ----------------------
# RoBERTa Model Loading and Prediction
# ----------------------

# Load the saved RoBERTa model using the correct path
roberta_model = ClassificationModel('roberta', output_dir, use_cuda=False)

# Real-world input text
real_world_text = ["This is a great product!", "I didn't like the service."]

# Predict the class
predictions_roberta, _ = roberta_model.predict(real_world_text)

print(f"RoBERTa Predictions: {predictions_roberta}")

# prediction on real-world input

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)
    txt = txt.strip()
    return txt

# load saved models; use use_cuda false if no gpu
# Changed 'bert_model' to './bert_best_model' to load the saved model from the correct directory
bert_model = ClassificationModel("bert", "./bert_best_model", use_cuda=False)
# Changed 'roberta_model' to './roberta_best_model' to load the saved model from the correct directory
roberta_model = ClassificationModel("roberta", "./roberta_best_model", use_cuda=False)

# set label encoder classes manually
encoder = LabelEncoder()
encoder.classes_ = np.array(['Bangalore','3935 sq. ft','Rooftop access','Old house','gym'])

input_text = input("enter product description: ")
cleaned_text = clean_text(input_text)

bert_pred, _ = bert_model.predict([cleaned_text])
roberta_pred, _ = roberta_model.predict([cleaned_text])

bert_lbl = encoder.inverse_transform(np.array(bert_pred))[0]
roberta_lbl = encoder.inverse_transform(np.array(roberta_pred))[0]

print("\nbert predicted category:", bert_lbl)
print("roberta predicted category:", roberta_lbl)

# prediction on real-world input

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zA-Z\s]', '', txt)
    txt = txt.strip()
    return txt



# set label encoder classes manually
encoder = LabelEncoder()
encoder.classes_ = np.array(['Bangalore','3935 sq. ft','Rooftop access','Old house','gym'])

input_text = input("enter product description: ")
cleaned_text = clean_text(input_text)

bert_pred, _ = bert_model.predict([cleaned_text])
roberta_pred, _ = roberta_model.predict([cleaned_text])

bert_lbl = encoder.inverse_transform(np.array(bert_pred))[0]
roberta_lbl = encoder.inverse_transform(np.array(roberta_pred))[0]

print("\nbert predicted category:", bert_lbl)
print("roberta predicted category:", roberta_lbl)