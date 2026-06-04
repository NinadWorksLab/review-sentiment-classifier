# Review-Sentiment-Classifier
Machine learning review sentiment classifier model using TF-IDF and Logistic Regression for customer review classification (positive/negative).

Overview-
This project uses TF-IDF Vectorization and Logistic Regression to predict sentiment. It can be applied to product reviews, hotel reviews, restaurant reviews, and other customer feedback datasets.

Feature-
Text preprocessing and cleaning,
TF-IDF feature extraction,
Logistic Regression classifier,
Sentiment prediction for new reviews,
Model evaluation using accuracy_score,f1_score,classification_report,
Easy-to-use prediction pipeline

Technologies Used-
Python,
Pandas,
Scikit-learn,
TF-IDF vector

Project Structure-
This project is currently implemented in a single Python file for simplicity.
review-sentiment-classifier/sentiment_analysis.py, README.md.

Installation-
Clone the repository:
git clone https://github.com/NinadWorksLab/review-sentiment-classifier.git
cd review-sentiment-classifier

Install the required packages:
pip install Pandas,Scikit-learn

Run the application:
python sentiment_analysis.py

Usage-
1. Run the Python script.
python review_sentiment_classifier.py
2. Upload a CSV or TXT file containing reviews.
3. Enter column name, which contain text review
4. The model will process the file and predict the sentiment of each review.

Example-
Enter review column: review 
                                        review     prediction
1.  0   Great customer service and nice packaging. -  Positive
2.  1    Excellent experience, highly recommended. -  Positive
3.  2            Amazing product, works perfectly! -  Positive
4.  3   The product stopped working after one day. -  Negative
5.  4   Terrible experience, not worth the money.  - Negative
6.  5          Very disappointed with the quality. -  Negative

Model Performance-
The model was trained using TF-IDF features and Logistic Regression and achieved strong performance on the test dataset.

Metrics evaluated:
1. Accuracy- 93.1%
2. Precision- 92.37%
3. Recall- 93.95%
4. F1-Score- 93.15%

Future Improvements-
1. Add complaint category classification to identify issues such as delivery delays, packaging problems, customer service complaints, and product quality defects.
2. Improve accuracy using advanced NLP and deep learning models.
3. Deploy the project as a web application.
4. Generate detailed complaint analysis reports.
5. Support additional file formats such as Excel (.xlsx) and JSON.
   
Author- 
NinadWorksLab

License- 
This project is licensed under the MIT License.









