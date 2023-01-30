import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("ratings.csv")

# Create a pivot table with users as rows and items as columns
pivot_table = df.pivot_table(index='user_id', columns='item_id', values='rating')

# Replace missing values with 0
pivot_table = pivot_table.fillna(0)

# Compute cosine similarity between all items
item_similarity = cosine_similarity(pivot_table.T)

# Create a dataframe with the similarity scores
item_similarity_df = pd.DataFrame(item_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Make predictions for each user-item combination
predictions = pivot_table.dot(item_similarity_df) / np.array([np.abs(item_similarity_df).sum(axis=1)])

# Replace the original ratings with the predicted ratings
pivot_table = pivot_table.values
predictions = predictions.values

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(pivot_table, predictions))
print("RMSE: ", rmse)
