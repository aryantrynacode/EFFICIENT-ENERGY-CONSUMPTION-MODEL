import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title (' 🔧 Predictive Energy Consumption Model for Steel Industry')

st.write('This project involves building a **predictive model** to estimate **energy consumption** in a smart small-scale steel industry using **multiple linear regression**, a statistical algorithm.')

st.title('📂 Dataset Source')

st.write('Dataset: Steel Industry Energy Consumption Dataset')

st.write('Collected From: A smart small-scale steel industry in South Korea')

st.link_button("View Dataset Source", "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption")

st.title('🎯 Objective')

st.write('To predict the energy consumption (Usage_kWh) using multiple process-related and temporal features. We aim to Develop a multiple linear regression (MLR) model to forecast energy usage')

df = pd.read_csv('C:/Users/xryan/OneDrive/Documents/streamlit2323/dataset/Steel_industry_data.csv')

st.subheader("📊 Dataset Overview")
st.write(df.head())

st.write("**Shape:**", df.shape)
st.write("**Columns:**", df.columns.tolist())

if st.checkbox("Show descriptive statistics"):
    st.write(df.describe())

df.drop(columns=['date','Load_Type','Day_of_week','WeekStatus'],inplace=True)

st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

target = st.selectbox("Select Target Variable", df.columns)
features = st.multiselect("Select Features", [col for col in df.columns if col != target])

if len(features) == 0:
    st.warning("Select at least one feature.")
    st.stop()

X = df.drop(columns=['Usage_kWh'])
y = df['Usage_kWh']
feature_columns = X.columns.tolist()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f'mean squared error: {mse}')
st.write(f'R-squared: {r2:.2f}')

st.subheader("🔍 Predict Energy Usage from Custom Input")

custom_input = {}
for feature in feature_columns:  # <- This should be same as X.columns
    custom_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([custom_input])
    prediction = model.predict(input_df)
    st.success(f"Predicted Energy Usage: {prediction[0]:.2f} kWh")


st.subheader("📌 Feature Correlation with Usage_kWh")


fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    df.corr()[['Usage_kWh']].sort_values(by='Usage_kWh', ascending=False),
    annot=True,
    cmap='coolwarm',
    ax=ax
)
ax.set_title("Correlation of Features with Usage_kWh")

st.pyplot(fig)

st.subheader("📉 Actual vs. Predicted Power Consumption")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_xlabel('Actual Power Consumption')
ax.set_ylabel('Predicted Power Consumption')
ax.set_title('Actual vs. Predicted Power Consumption')
ax.grid(True)

st.pyplot(fig)
st.markdown("### ✅ Insights and Conclusion")
st.markdown("""
- The model achieved an **R² of 0.98**, which indicates good performance.
- Most influential features based on correlation were:
    - CO2 Emissions
    - Power Factor
    - Lagging and Leading Reactive Power
- This model can help optimize energy usage in smart manufacturing.
""".format(r2_score(y_test, y_pred)))

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
st.markdown('Thank you for checking this blog ~ By Aryan')









