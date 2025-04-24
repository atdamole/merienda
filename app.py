import gradio as gr
import joblib
import numpy as np

# Load models
model_flavor = joblib.load("dtc_model_flavor.pkl")
model_topping = joblib.load("dtc_model_topping.pkl")
model_drink = joblib.load("dtc_model_drink.pkl")

# Load encoders
encoder_flavor = joblib.load("encoder_flavor.pkl")
encoder_topping = joblib.load("encoder_topping.pkl")
encoder_drink = joblib.load("encoder_drink.pkl")
input_encoders = joblib.load("input_encoders.pkl")

# Prediction function
def predict_merienda(mood, weather, craving_level, last_meal, budget):
    features = [mood, weather, craving_level, last_meal, budget]
    encoded = [input_encoders[col].transform([val])[0] for col, val in zip(input_encoders.keys(), features)]
    encoded_np = np.array(encoded).reshape(1,-1)

    pred_flavor = encoder_flavor.inverse_transform(model_flavor.predict(encoded_np))[0]
    pred_topping = encoder_topping.inverse_transform(model_topping.predict(encoded_np))[0]
    pred_drink = encoder_drink.inverse_transform(model_drink.predict(encoded_np))[0]
    return pred_flavor, pred_topping, pred_drink

# Dropdown options
mood_list = input_encoders["mood"].classes_.tolist()
weather_list = input_encoders["weather"].classes_.tolist()
craving_list = input_encoders["craving_level"].classes_.tolist()
last_meal_list = input_encoders["last_meal"].classes_.tolist()
budget_list = input_encoders["budget"].classes_.tolist()

# Gradio Interface
iface = gr.Interface(
    fn=predict_merienda,
    inputs=[
        gr.Dropdown(mood_list, label="🧠 Mood"),
        gr.Dropdown(weather_list, label="🌦️ Weather"),
        gr.Dropdown(craving_list, label="🔥 Craving Level"),
        gr.Dropdown(last_meal_list, label="🍽️ Last Meal"),
        gr.Dropdown(budget_list, label="💸 Budget"),
    ],
    outputs=[
        gr.Textbox(label="✨ Flavor Match"),
        gr.Textbox(label="🍳 Topping Pairing"),
        gr.Textbox(label="🥤 Drink Suggestion"),
    ],
    title="🥡 Merienda Matchmaker",
    description="""
🎉 Hungry? Let’s match your cravings with the perfect pancit canton combo!
Pick your mood, weather, and vibe — we’ll do the rest. 🧠🍜
""",
    theme="soft",  # optional, Gradio built-in themes
)
 
iface.launch()