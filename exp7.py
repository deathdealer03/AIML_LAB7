import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Creating a sample dataset
data = pd.read_csv('final.csv')

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define the features and target variable (if applicable)
X = df[['Age', 'Dailyrate', 'Department']]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Dailyrate']),
        ('cat', OneHotEncoder(), ['Department'])
    ]
)

# Create a pipeline that includes preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
X_processed = pipeline.fit_transform(X)

# Convert the result back to a DataFrame
column_names = (
    pipeline.named_steps['preprocessor']
        .transformers_[0][1].get_feature_names_out(['Age', 'Dailyrate']).tolist() + 
    list(pipeline.named_steps['preprocessor']
        .transformers_[1][1].get_feature_names_out(['Department']))
)

df_processed = pd.DataFrame(X_processed, columns=column_names)

# Fit and transform the data
X_processed = pipeline.fit_transform(X)

# Convert the result back to a DataFrame
column_names = (pipeline.named_steps['preprocessor']
                .transformers_[0][1].get_feature_names_out(['Age', 'Dailyrate']).tolist() +
                list(pipeline.named_steps['preprocessor'].transformers_[1][1]
                .get_feature_names_out(['Department'])))

df_processed = pd.DataFrame(X_processed, columns=column_names)

print("\nProcessed DataFrame:")
print(df_processed)

# Plotting the results
# Plot original data
plt.figure(figsize=(12, 6))

# Original Age vs Dailyrate
plt.subplot(1, 2, 1)
plt.scatter(df['Age'], df['Dailyrate'], color='blue')
plt.title('Original Data: Age vs Dailyrate')
plt.xlabel('Age')
plt.ylabel('Dailyrate')

# Processed Age vs Dailyrate
plt.subplot(1, 2, 2)
plt.scatter(df_processed['Age'], df_processed['Dailyrate'], color='orange')
plt.title('Processed Data: Age vs Dailyrate (Scaled)')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Dailyrate')

plt.tight_layout()
plt.show()