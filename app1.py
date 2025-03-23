import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

# Title and description
st.write("## PERSONAL FITNESS TRACKER...The Body Burner: Your Fitness, Your Formula")
st.write("""Welcome to Your Personal Fitness Tracker! 🏃‍♂️💪
Track Your Fitness Journey with Data-Driven Insights.
This app is designed to empower you on your fitness journey by providing personalized insights into how your body burns calories during exercise. By entering your personal parameters such as Age, Gender, BMI, Heart Rate, and Exercise Duration, you will receive accurate predictions about the calories burned during your workout. But that's not all!""")

st.sidebar.header("User Input Parameters: ")

# Function to collect user input
def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Calculate BMI before splitting
exercise["BMI"] = exercise["Weight"] / ((exercise["Height"] / 100) ** 2)
exercise["BMI"] = round(exercise["BMI"], 2)

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# ---- RANDOM FOREST REGRESSOR WITHOUT GRIDSEARCHCV ----
random_reg = RandomForestRegressor(n_estimators=1000)  # Default Random Forest Regressor without additional hyperparameters
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

# ---- FEATURE IMPORTANCE VISUALIZATION ----
feature_importances = random_reg.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
feature_df.plot(kind='bar', x='Feature', y='Importance', ax=ax, color='skyblue')
plt.title('Feature Importance')
st.pyplot(fig)

# ---- ADDING GOAL SETTING AND TRACKING ----
def set_goals():
    goal_calories = st.sidebar.number_input("Set Your Target Calories Burned:", min_value=0, max_value=1000, value=500)
    goal_duration = st.sidebar.number_input("Set Your Target Exercise Duration (min):", min_value=0, max_value=120, value=30)
    return goal_calories, goal_duration

goal_calories, goal_duration = set_goals()

# Calculate progress
calories_burned = prediction[0]
calories_progress = (calories_burned / goal_calories) * 100
duration_progress = (df["Duration"].values[0] / goal_duration) * 100

# Display progress
st.header("Your Fitness Goal Progress")
st.write(f"**Calories Burned Today**: {calories_burned} kcal")
st.write(f"**Your Goal**: {goal_calories} kcal")
st.write(f"**Progress towards your goal**: {calories_progress:.2f}%")

st.write(f"**Exercise Duration Today**: {df['Duration'].values[0]} min")
st.write(f"**Your Goal**: {goal_duration} min")
st.write(f"**Progress towards your goal**: {duration_progress:.2f}%")

# ---- GAUGE CHART FOR PROGRESS TRACKING ----
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=calories_progress,
    title={"text": "Calories Progress"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "darkblue"},
        "steps": [
            {"range": [0, 50], "color": "lightgray"},
            {"range": [50, 100], "color": "skyblue"},
        ],
    },
))
st.plotly_chart(fig)

# ---- INTERACTIVE SCATTER PLOTS WITH REGRESSION LINE ----
st.write("### Interactive Scatter Plots")
features_to_plot = ["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]

for feature in features_to_plot:
    fig = px.scatter(exercise_df, x=feature, y="Calories", title=f"{feature} vs Calories Burned", labels={feature: feature, 'Calories': 'Calories Burned'})
    fig.update_layout(showlegend=True)
    
    # Add a regression line
    x_vals = np.array(exercise_df[feature])
    y_vals = np.array(exercise_df['Calories'])
    coefficients = np.polyfit(x_vals, y_vals, 1)
    polynomial = np.poly1d(coefficients)
    line_x = np.linspace(x_vals.min(), x_vals.max(), 100)
    line_y = polynomial(line_x)
    
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Regression Line', line=dict(color='red')))
    st.plotly_chart(fig)
    
    st.write(f"""
        **Explanation of {feature} vs Calories Burned:**
        - A **positive correlation** suggests that as **{feature}** increases, calories burned increases.
        - A **negative correlation** suggests that as **{feature}** increases, calories burned decreases.
    """)

# ---- INTERACTIVE CORRELATION HEATMAP ----
st.write("### Correlation Heatmap of Features with Calories Burned")
corr = exercise_df[["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]].corr()

fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(corr.columns),
    y=list(corr.index),
    colorscale='Viridis'
)
st.plotly_chart(fig)

# ---- GENERAL INFORMATION AND CONCLUSION BASED ON USER INPUT ----
st.write("### Conclusion: Your Personal Fitness Overview")

if df["BMI"].values[0] > 30:
    st.write("**You have a high BMI, which suggests that you might have some extra body fat.** Consider focusing on cardiovascular exercises to burn more calories.")
else:
    st.write("**Your BMI is within a normal range, indicating a healthy body composition.** Keep maintaining a balanced diet and regular exercise routine.")

if df["Heart_Rate"].values[0] > 100:
    st.write("**Your heart rate is relatively high during exercise.** This indicates that you might be pushing yourself hard during physical activities.")
else:
    st.write("**Your heart rate during exercise is within a normal range.** This suggests you're engaging in moderate exercise intensity.")

if df["Duration"].values[0] > 30:
    st.write("**You're exercising for more than 30 minutes, which is fantastic!** Consistent exercise duration helps build endurance and aids in long-term calorie burning.")
else:
    st.write("**You might want to consider extending your exercise duration slightly.** Gradually increasing your workout time will help you burn more calories.")

st.write("---")
st.write("Thank you for using our Personal Fitness Tracker! Stay consistent and keep track of your progress. Remember, fitness is a journey, not a destination! 💪")
