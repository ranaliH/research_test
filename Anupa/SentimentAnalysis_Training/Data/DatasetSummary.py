import pandas as pd

# Read the dataset into a pandas DataFrame
df2 = pd.read_csv(r"D:\SLIIT\Year 4\Semester 1\Research\ChatBot\Codes\SentimentAnalysis_Training\Data\test.csv")

# Count the number of data points for each class
class_counts = df2['category'].value_counts()

# Display the class counts
for class_name, count in class_counts.items():
    print("Class:", class_name)
    print("Frequency:", count)
    print()
