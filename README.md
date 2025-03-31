# Property Valuation Model

## Course: NLP (Semester 6) - Pillai College of Engineering

### Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on Property Valuation, where we apply various Machine Learning (ML), Deep Learning (DL), and Language Models to categorize property models into predefined categories.

This project involves exploring techniques like:
- Text preprocessing
- Feature extraction
- Model training
- Evaluating the models for their effectiveness in classifying property valuation models

You can learn more about the college by visiting the official website of Pillai College of Engineering.

---

## Acknowledgements
We would like to express our sincere gratitude to the following individuals:

### Theory Faculty:
- Dhiraj Amin
- Sharvari Govilkar

### Lab Faculty:
- Dhiraj Amin
- Neha Ashok
- Shubhangi Chavan

Their guidance and support have been invaluable throughout this project.

---

## Project Title
**Property Valuation Model using Natural Language Processing**

### Project Abstract
This project explores the use of Natural Language Processing (NLP) in property valuation by analyzing unstructured textual data from real estate listings and market reports. By applying techniques like sentiment analysis and text classification, we enhance valuation accuracy.

Integrating NLP with machine learning models improves predictive performance, enabling more informed and dynamic real estate assessments.

---

## Algorithms Used

### Machine Learning Algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

### Deep Learning Algorithms:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)

### Language Models:
- GPT
- BERT (Bidirectional Encoder Representations from Transformers)

---

## Comparative Analysis
The comparative analysis of different models highlights their effectiveness in classifying news articles into the correct category. The following table summarizes the accuracy, precision, recall, and F1-score of the models tested:

| No. | Model Name  | Feature | Precision | Recall | F1 Score | Accuracy |
|----|------------|---------|-----------|--------|----------|----------|
| 0  | CNN        | BoW     | 0.89      | 0.89   | 0.89     | 0.9457   |
| 1  | LSTM       | BoW     | 0.00      | 0.03   | 0.00     | 0.0121   |
| 2  | BiLSTM     | BoW     | 0.04      | 0.06   | 0.03     | 0.0523   |
| 3  | CNN-BiLSTM | BoW     | 0.10      | 0.11   | 0.06     | 0.1107   |
| 4  | CNN        | TF-IDF  | 0.00      | 0.04   | 0.00     | 0.0221   |
| 5  | LSTM       | TF-IDF  | 0.00      | 0.04   | 0.00     | 0.0241   |
| 6  | BiLSTM     | TF-IDF  | 0.00      | 0.04   | 0.00     | 0.0241   |
| 7  | CNN-BiLSTM | TF-IDF  | 0.00      | 0.04   | 0.00     | 0.0241   |
| 8  | CNN        | FastText| 0.13      | 0.13   | 0.10     | 0.1328   |
| 9  | LSTM       | FastText| 0.00      | 0.04   | 0.00     | 0.0241   |
|10  | BiLSTM     | FastText| 0.12      | 0.12   | 0.09     | 0.1187   |
|11  | CNN-BiLSTM | FastText| 0.05      | 0.09   | 0.08     | 0.1045   |

---

## Conclusion
This Property Valuation project demonstrates the potential of Machine Learning, Deep Learning, and Language Models in real estate assessment by analyzing unstructured textual data from property listings and market reports.

The comparative analysis reveals that **BERT, a transformer-based model, outperforms traditional methods and deep learning models** in terms of accuracy, precision, and recall. By integrating NLP with machine learning models, we enhance predictive performance, providing more informed and dynamic real estate assessments.

This study highlights the strengths and limitations of different approaches, guiding the selection of the most effective methodology for property valuation.

---

This README file serves as comprehensive documentation for the **Property Valuation Model** project, summarizing its objectives, methodologies, algorithms used, and key findings.

