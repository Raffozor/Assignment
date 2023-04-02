import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreProcessor:
        
    def __init__(self, data : pd.DataFrame):
        self.data = data
        """
        Initialize the DataPreProcessor class with a given pandas DataFrame.

        Args:
            data (pd.DataFrame): Input dataset to be preprocessed.
        """
    
    def replace_missing_values(self, missing_value):
        """
        Replace missing values in the dataset with NaN.

        Args:
            missing_value (any): Input value to be replaced by NaN.
        """
        for col in self.data.columns:
            self.data[col] = self.data[col].apply(lambda x: np.nan if x == missing_value else x)
    
    def training_test(self, training: float = 0.7):
        """
        Split the dataset into training and test subsets.

        Args:
            training (float): Proportion of data to be used for training. Default is 0.7.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Training and test subsets of the original dataset.
        """
        assert 0 <= training <= 1, "training must be a float between 0 and 1."
        
        training_data = self.data.sample(frac=training)
        test_data = pd.merge(training_data, self.data, how='outer', indicator=True)
        test_data = test_data[test_data['_merge'] == 'right_only']
        test_data = test_data.drop('_merge', axis=1)
        
        return training_data, test_data
        
    def summary_nan(self):
        """
        Plot the percentage of NaN values for each variable in the input dataset.

        Args:
            data (pd.DataFrame): Input dataset.
        """
        missing = self.data.isna().sum()
        missing_values = [i / self.data.shape[0] for i in missing]
        columns = list(self.data.columns)

        fig = plt.figure(figsize=(16*0.75, 9), dpi=80)
        plt.barh(columns, missing_values, height=0.7)
        plt.title(label='Missing Values', color='xkcd:pale red', fontsize=18, pad=13, fontweight='bold')
        plt.xlabel('% of NaN', color='xkcd:pale red', fontsize=14, fontweight='bold')
        plt.ylabel('Variable', color='xkcd:pale red', fontsize=14, fontweight='bold')
        plt.xticks(np.arange(0, 1.1, step=0.1), fontsize=10, color='xkcd:cadet blue')
        plt.yticks(fontsize=10, color='xkcd:cadet blue')
        plt.grid(color='black', linewidth=1, axis='both', alpha=0.5, which='major')
        plt.show()
    
    def fillna(self) -> None:
        """
        Fill missing values in the input dataset using the mean for numerical columns and mode for categorical columns.
        """
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            else:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
    
    def run_preprocessing(self, missing_value, training : float = 0.7):
        """
        Runs all the preprocessing steps in order and returns the training and test datasets.

        Args:
            missing_value (str): The string to be interpreted as missing values.
            training_ratio (float): The ratio of the data to be used for training (default 0.7).

        Returns:
            training (pd.DataFrame): The preprocessed training dataset.
            test (pd.DataFrame): The preprocessed test dataset.
        """
        assert 0 <= training <= 1, "training must be a float between 0 and 1."
        self.replace_missing_values(missing_value)
        self.fillna()
        training, test = self.training_test(training)
        return training, test
    