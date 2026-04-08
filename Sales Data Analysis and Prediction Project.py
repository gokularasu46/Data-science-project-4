# Sales Data Analysis and Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create Sales Dataset
data = {
    "Month":[1,2,3,4,5,6,7,8,9,10],
    "Sales":[200,220,250,300,320,350,370,400,420,450]
}

df = pd.DataFrame(data)

print("Sales Dataset")
print(df)

# Step 2: Data Analysis
print("\nBasic Sales Analysis")
print("Average Sales:", df["Sales"].mean())
print("Maximum Sales:", df["Sales"].max())
print("Minimum Sales:", df["Sales"].min())

# Step 3: Data Visualization
plt.plot(df["Month"], df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Step 4: Prepare Data for Machine Learning
X = df[["Month"]]
y = df["Sales"]

# Step 5: Train Machine Learning Model
model = LinearRegression()
model.fit(X,y)

# Step 6: Predict Future Sales
month = int(input("\nEnter future month number to predict sales: "))
prediction = model.predict([[month]])

print("Predicted Sales for Month",month,":",prediction[0])

# Step 7: Show Prediction Graph
plt.scatter(df["Month"], df["Sales"], color="blue")
plt.plot(df["Month"], model.predict(X), color="red")

plt.title("Sales Prediction Model")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()