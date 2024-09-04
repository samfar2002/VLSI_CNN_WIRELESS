import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """ Load the dataset from a CSV file. """
    return pd.read_csv(filepath)

def plot_confusion_matrix(cm, fold_idx):
    """ Plot and save the confusion matrix for a given fold. """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', cbar=False)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix - Fold {}'.format(fold_idx))
    # Save the plot as a PNG file
    plt.savefig('Confusion_Matrix_Fold_{}.png'.format(fold_idx))
    plt.close()  # Close the figure to free up memory

def main():
    # Load the data
    df = load_data('extracted_features_with_labels.csv')
    
    # Prepare data
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Set up k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=False)
    fold_idx = 1
    
    # Iterate through each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the logistic regression model
        # model = LogisticRegression(max_iter=1000,class_weight='balanced')
        model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')  # Specified n_estimators and random_state
        model.fit(X_train, y_train)
        
        # Predict the test set
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print classification report for the current fold
        print("Classification Report for Fold {}:".format(fold_idx))
        print(classification_report(y_test, y_pred))
        
        # Plot and save confusion matrix
        plot_confusion_matrix(cm, fold_idx)
        
        fold_idx += 1

if __name__ == '__main__':
    main()
