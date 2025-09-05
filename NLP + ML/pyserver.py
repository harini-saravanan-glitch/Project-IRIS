from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load model & tokenizer
try:
    print("Loading model and tokenizer...")
    output_dir = './model_save/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(output_dir)
    model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}. Make sure './model_save/' exists and contains a trained model.")
    model = None
    tokenizer = None
    device = torch.device("cpu")

def predict_grooming_intent(message: str):
    if not model or not tokenizer:
        return {"error": "Model not loaded"}, 500

    model.eval()
    encoded_review = tokenizer.encode_plus(
        message,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    prediction = torch.argmax(logits, dim=1).item()
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence = probs[0][prediction].item()

    label = "Predatory Intent Detected" if prediction == 1 else "Normal"
    return {"prediction": label, "confidence": confidence}

@app.route("/")
def home():
    return jsonify({"message": "Flask server is running. Use POST /predict with JSON {'text': 'your message'}"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' field is required."}), 400
    
    message = data['text']
    try:
        result = predict_grooming_intent(message)
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)


