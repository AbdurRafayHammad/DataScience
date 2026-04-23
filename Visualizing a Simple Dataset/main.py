import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("iris")

# Show first rows
print(df.head())

# Structure
print("Shape:", df.shape)
print("Columns:", df.columns)

print(df.info())
print(df.describe())

sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
plt.title("Sepal Length vs Width")
plt.show()
sns.histplot(df["petal_length"], kde=True)
plt.title("Petal Length Distribution")
plt.show()

sns.boxplot(x="species", y="sepal_length", data=df)
plt.title("Sepal Length by Species")
plt.show()