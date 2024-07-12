# URL Classification : Productive or Non-Productive

## Overview
This project aims to classify URLs into two categories: productive and non-productive based on their web content. We'll use Python for the implementation.

## Table of Contents
1. Introduction
2. Dataset
3. Data Preprocessing
4. Feature Extraction
5. Model Building
6. Evaluation
7. Conclusion

## Introduction
When a company classifies URLs in terms of productivity, it aims to distinguish between websites that contribute positively to work-related tasks and those that do not. Here’s why such classification is needed:

#### 1. Productivity Monitoring:
Companies want to ensure that employees spend their time efficiently during work hours.
By classifying URLs, they can identify which websites are directly related to work (e.g., project management tools, research resources) and which are not (e.g., social media, entertainment sites).
Monitoring productivity helps optimize employee performance and resource allocation.

#### 2. Resource Allocation and Bandwidth Management:
Companies need to allocate network resources effectively.
By categorizing URLs, they can prioritize bandwidth for work-related sites and limit access to non-productive ones.
This ensures that critical business applications receive sufficient resources while minimizing distractions.

#### 3. Security and Acceptable Use Policies:
URL classification helps enforce acceptable use policies.
Companies can define rules based on productivity categories (e.g., allow access to educational sites, restrict gaming sites).
It prevents security risks (e.g., malware from non-productive sites) and maintains a secure work environment.

#### 4. Insight for Decision-Making:
Managers can analyze trends in URL access.
Are employees spending excessive time on non-productive sites? Are there productivity bottlenecks?
Data-driven decisions can improve overall efficiency and employee satisfaction.

In summary, URL classification for productivity allows companies to strike a balance between work-related browsing and potential distractions, leading to better resource management and a more focused workforce. 


## Dataset
For this project, a self-created dataset is used, but you can procure any from Kaggle itself, just remember to replace the labels in the code accordingly.  
Sample dataset used:
![image](https://github.com/sarkar-sayan/URL-Classification/assets/105176992/1b80ea00-1c4a-4081-a961-7c526dd66369)

## Data Preprocessing
Read the [repo_logs](https://github.com/sarkar-sayan/URL-Classification/blob/main/repo_logs) file first for better understanding.
##### 1. get_metadata_from_url(url):
This function retrieves metadata (such as title, description, image, and text) from a given URL.  
It uses the requests library to fetch the webpage content and BeautifulSoup for HTML parsing.  
If an error occurs during the request (e.g., invalid URL), it returns an error message.  
The extracted metadata is returned as a dictionary.  
##### 2. preprocess_text(text):
This function preprocesses text by:  
Converting it to lowercase.  
Removing punctuation.  
Splitting it into tokens (words).  
Filtering out stop words (common words like “the,” “and,” etc.).  
The cleaned text is returned.  
##### 3. translate_text_if_needed(text):
If the input text is not in English, this function attempts to translate it to English using the detect and translator libraries.  
If translation fails or the text is already in English, it returns the original text.  
##### 4. preprocess_metadata(metadata):
Combines the cleaned title, description, and text from the metadata.  
Translates them to English if needed.  
Returns the combined and cleaned content.  
##### 5. extract_domain(url):
Extracts the domain (e.g., “example.com”) from a given URL.  
##### 6. prepare_dataset(productive_keywords, non_productive_keywords, dataset_url):
Reads a CSV dataset from the specified URL.  
Keeps only the ‘url’ and ‘label’ columns.  
Scrapes metadata for each URL using get_metadata_from_url.  
Applies preprocessing to the metadata using preprocess_metadata.  
Extracts the domain from each URL.  
Calculates the count of productive and non-productive keywords in the cleaned content.  

## Feature Extraction
##### 1. create_feature_matrix(df):
This function creates a feature matrix from a DataFrame (df).  
It performs the following steps:  
Uses a vectorizer (which is not defined in the snippet) to transform the ‘clean_content’ column into a TF-IDF (Term Frequency-Inverse Document Frequency) matrix.  
Extracts the ‘productive_keyword_count’ and ‘non_productive_keyword_count’ columns as keyword counts.  
Combines the TF-IDF matrix and keyword counts horizontally (using np.hstack()).  
The resulting feature matrix (X) contains both textual features (TF-IDF) and keyword counts.  
##### 2. Usage:
The snippet ends with X = create_feature_matrix(data), where data is presumably a DataFrame containing relevant columns (‘clean_content’, ‘productive_keyword_count’, ‘non_productive_keyword_count’).  
The resulting feature matrix X can be used for further analysis or modeling.  
Remember to define the vectorizer before using this code snippet, and ensure that your data DataFrame contains the necessary columns.  

## Model Building
#### 1. Train-Test Split:
The data is split into training and testing sets using train_test_split.  
The training set (X_train, y_train) will be used to train the model, and the testing set (X_test, y_test) will be used for evaluation.  
I've set a test size of 40% and a random seed for reproducibility.  
#### 2. Model Training:
I've chosen the MultinomialNB (Naive Bayes) classifier for my model.  
The model is trained using the training data (X_train, y_train).  

## Evaluation
I’ve made predictions on the test set using model.predict(X_test).  
The accuracy of the model is calculated using accuracy_score(y_test, y_pred).  
The printed output shows the accuracy, precision, recall, F1-score, and support for each class (productive and non-productive).  
#### Interpretation:
An accuracy of 1.0 indicates that the model perfectly predicts the test data.  
The precision, recall, and F1-score are also 1.0 for both classes, suggesting excellent performance.  

## Conclusion
Multinomial Naive Bayes is a probabilistic classifier to calculate the probability distribution of text data, which makes it well-suited for data with features that represent discrete frequencies or counts of events in various natural language processing (NLP) tasks.  
The term “multinomial” refers to the type of data distribution assumed by the model.  The features in text classification are typically word counts or term frequencies. The multinomial distribution is used to estimate the likelihood of seeing a specific set of word counts in a document.  
Also it works way better in smaller datasets as it assumes some features beforehand (being based on Bayes Theorem of Probability)  


---

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
