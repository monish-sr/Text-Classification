# Text Classification
This project classifies messages as **spam** or **ham** using models like Logistic Regression, KNN, Naive Bayes, and Random Forest.

## Project Overview
This project focuses on text classification for identifying whether a message is spam or ham (not spam). The approach involves preprocessing textual data, converting it into numerical representations, and training multiple machine learning models to predict the message type. The models used in this project include Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, and Random Forest Classifier.

## Dataset
This project utilizes the Mail dataset, which includes spam mails and non spam(Ham) mails. This dataset contains 31 rows and 2 columns.

## Methodology
1. **Data Collection**:  
   - Used the dataset `Spam or Ham.csv`, containing messages and their corresponding labels (spam or ham).

2. **Data Preprocessing**:  
   - Checked for null values and inconsistencies.  
   - Encoded labels (`Type`) into binary format (0 for ham, 1 for spam).  
   - Cleaned text: converted to lowercase, removed digits, special characters, and stopwords.  
   - Applied lemmatization for word normalization.  

3. **Feature Extraction**:  
   - Transformed the preprocessed text into numerical vectors using **TF-IDF vectorization**.

4. **Data Splitting**:  
   - Split the dataset into training (70%) and testing (30%) sets.

5. **Model Training**:  
   - Trained the following machine learning models on the training set:  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Naive Bayes  
     - Random Forest Classifier  

6. **Model Evaluation**:  
   - Predicted results on the test set for each model.  
   - Evaluated performance using:  
     - **Accuracy**  
     - **Confusion Matrix**  
     - **Classification Report**  

7. **Comparison and Insights**:  
   - Compared model performances to identify the best-performing algorithm for spam detection.

## Results  
1. **Logistic Regression**:  
   - **Accuracy**: High  
   - **Confusion Matrix**: Balanced detection of spam and ham.  
   - **Classification Report**: Strong precision, recall, and F1-score.  

2. **K-Nearest Neighbors (KNN)**:  
   - **Accuracy**: Moderate  
   - **Confusion Matrix**: Lower spam detection accuracy.  
   - **Classification Report**: Precision and recall slightly lower compared to Logistic Regression.

3. **Naive Bayes**:  
   - **Accuracy**: High  
   - **Confusion Matrix**: Excellent spam detection with a slight trade-off in false positives.  
   - **Classification Report**: High recall and F1-score for spam.

4. **Random Forest Classifier**:  
   - **Accuracy**: Very High  
   - **Confusion Matrix**: Effective at distinguishing spam and ham.  
   - **Classification Report**: Best overall precision, recall, and F1-score.

### **Key Insight**  
- **Random Forest** outperformed other models with the highest accuracy and robust spam detection.  
- **Logistic Regression** and **Naive Bayes** also showed competitive results, while **KNN** was less effective for this task.
