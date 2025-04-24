import gradio as gr
import joblib
import numpy as np

# ✅ Use Gradio's caching decorator to avoid reloading files on every run
@gr.cache()
def load_models_and_encoders():
    models = {
        "flavor": joblib.load("dtc_model_flavor.pkl"),
        "topping": joblib.load("dtc_model_topping.pkl"),
        "drink": joblib.load("dtc_model_drink.pkl")
    }
    encoders = {
        "flavor": joblib.load("encoder_flavor.pkl"),
        "topping": joblib.load("encoder_topping.pkl"),
        "drink": joblib.load("encoder_drink.pkl"),
        "inputs": joblib.load("input_encoders.pkl")
    }
    return models, encoders

models, encoders = load_models_and_encoders()

# Dropdown options
mood_list = encoders["inputs"]["mood"].classes_.tolist()
weather_list = encoders["inputs"]["weather"].classes_.tolist()
craving_list = encoders["inputs"]["craving_level"].classes_.tolist()
last_meal_list = encoders["inputs"]["last_meal"].classes_.tolist()
budget_list = encoders["inputs"]["budget"].classes_.tolist()

# ✅ Prediction function only runs after input
def predict_merienda(mood, weather, craving_level, last_meal, budget):
    features = [mood, weather, craving_level, last_meal, budget]
    encoded = [encoders["inputs"][col].transform([val])[0] for col, val in zip(encoders["inputs"].keys(), features)]
    encoded_np = np.array(encoded).reshape(1, -1)

    pred_flavor = encoders["flavor"].inverse_transform(models["flavor"].predict(encoded_np))[0]
    pred_topping = encoders["topping"].inverse_transform(models["topping"].predict(encoded_np))[0]
    pred_drink = encoders["drink"].inverse_transform(models["drink"].predict(encoded_np))[0]

    return pred_flavor, pred_topping, pred_drink

# 🌈 Gradio Interface with emoji labels and soft theme
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
    theme="soft",
)

iface.launch()
