from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer, util
import time
import random
import json
import threading  

app = Flask(__name__)
socketio = SocketIO(app)

# Load the language model and embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# Load the fine-tuned model and tokenizer
intent_model = BertForSequenceClassification.from_pretrained(r'C:\Python Work\PROJECT\fine-tuned-bert')
tokenizer = BertTokenizer.from_pretrained(r'C:\Python Work\PROJECT\fine-tuned-bert')

# Intent classification pipeline
nlp_pipeline = pipeline("text-classification", model=intent_model, tokenizer=tokenizer)

# Function to classify intent using BERT model
def classify_intent(text):
    result = nlp_pipeline(text)[0]
    intent_label = result['label']
    intent = 0 if intent_label == 'LABEL_0' else 1
    return intent

# Load emergency data from JSON file with error handling
try:
    with open('emergency_data.json', 'r') as f:
        emergency_data = json.load(f).get('emergencies', [])
except FileNotFoundError:
    print("Error: 'emergency_data.json' file not found.")
    emergency_data = []
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from 'emergency_data.json'.")
    emergency_data = []

# Generate embeddings for emergency keywords
emergency_embeddings = []
for emergency in emergency_data:
    keywords_text = " ".join(emergency.get('keywords', []))
    emergency_embeddings.append(embedding_model.encode(keywords_text))

# Track conversation state
conversation_state = {}

def find_best_match(user_input):
    user_embedding = embedding_model.encode(user_input)
    similarities = [util.cos_sim(user_embedding, embedding)[0][0].item() for embedding in emergency_embeddings]
    best_match_idx = similarities.index(max(similarities))
    return emergency_data[best_match_idx], max(similarities)

def delayed_response(session_id):
    time.sleep(15)  # Simulating delay
    state = conversation_state.get(session_id)
    best_match = state.get('best_match')
    socketio.emit('bot_response', {"message": f"I found a match: {best_match['name']}. {best_match['steps']} Dr. Adrin is on the way."}, room=session_id)
    socketio.emit('bot_response', {"message": "Don't worry, please follow these steps, Dr. Adrin will be with you shortly."}, room=session_id)
    del conversation_state[session_id]  # Reset conversation state for the session

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_message(data):
    user_input = data['message'].lower()
    session_id = data.get('session_id')  # Assuming you track session_id for different users
    
    if session_id not in conversation_state:
        conversation_state[session_id] = {'step': 1, 'waiting': False, 'best_match': None, 'last_question': ''}

    state = conversation_state[session_id]

    def repeat_last_question():
        emit('bot_response', {"message": f"I don’t understand that. {state['last_question']}"})

    if state['step'] == 1:
        state['last_question'] = "Are you having an emergency, or would you like to leave a message?"
        emit('bot_response', {"message": state['last_question']})
        state['step'] = 2
    
    elif state['step'] == 2:
        intent = classify_intent(user_input)
        
        if intent == 0:  # Assuming LABEL_0 corresponds to "emergency"
            state['last_question'] = "Please describe the emergency."
            emit('bot_response', {"message": "It appears that you are in distress."})
            time.sleep(3)
            emit('bot_response', {"message": state['last_question']})
            state['step'] = 3
        
        elif intent == 1:  # Assuming LABEL_1 corresponds to "message"
            state['last_question'] = "Please leave a message. I will make sure he receives it."
            emit('bot_response', {"message": "It appears that you wish to talk to the doctor."})
            time.sleep(3)
            emit('bot_response', {"message": state['last_question']})
            state['step'] = 6
        
        else:
            repeat_last_question()

    elif state['step'] == 3:
        if emergency_data:
            best_match, similarity_score = find_best_match(user_input)

            # Setting a lower threshold and more explicit matching mechanism
            if similarity_score < 0.5:
                state['last_question'] = "Could you please describe the situation in more detail?"
                emit('bot_response', {"message": "I'm not sure I understand."})
                emit('bot_response', {"message": state['last_question']})
            else:
                state['best_match'] = best_match
                state['last_question'] = "Can you tell me which area you are located right now?"
                emit('bot_response', {"message": "I am checking what you should do immediately."})
                emit('bot_response', {"message": state['last_question']})
                state['step'] = 4
        else:
            emit('bot_response', {"message": "Emergency data is not available. Please try again later."})

    elif state['step'] == 4:
        state['location'] = user_input
        eta = random.randint(5, 15)
        emit('bot_response', {"message": f"Dr. Adrin will arrive at {state['location']} in approximately {eta} minutes."})
        state['last_question'] = "Please hold on while I check the database for the next steps."
        emit('bot_response', {"message": state['last_question']})
        state['step'] = 5

        # Start the delayed response in a new thread
        threading.Thread(target=delayed_response, args=(session_id,)).start()

    elif state['step'] == 5:
        if "too late" in user_input or "late" in user_input:
            if not state['waiting']:
                emit('bot_response', {"message": "I understand that you are worried that Dr. Adrin will arrive too late. Please wait a moment while I check the best course of action."})
                state['waiting'] = True  # Mark that the user has expressed concern and the bot has responded
            else:
                if 'best_match' in state:
                    emit('bot_response', {"message": f"Meanwhile, we would suggest that you start {state['best_match']['name']}. {state['best_match']['steps']}."})
                    state['step'] = 6  # Move to step 6 after providing the response

        else:
            repeat_last_question()
    
    elif state['step'] == 6:
        emit('bot_response', {"message": "Thanks for the message, I will forward it to Dr. Adrin."})
        del conversation_state[session_id]  # Reset conversation state for the session


if __name__ == "__main__":
    socketio.run(app, debug=True)
