from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer, util
import time
import random
import json
import threading  
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

app = Flask(__name__)
socketio = SocketIO(app)

# Load the language model and embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the fine-tuned model and tokenizer for intent classification
intent_model = BertForSequenceClassification.from_pretrained(r'C:\Python Work\PROJECT\fine-tuned-bert')
tokenizer = BertTokenizer.from_pretrained(r'C:\Python Work\PROJECT\fine-tuned-bert')

# Intent classification pipeline
nlp_pipeline = pipeline("text-classification", model=intent_model, tokenizer=tokenizer)

# Function to classify intent using the BERT model
def classify_intent(text):
    result = nlp_pipeline(text)[0]
    intent_label = result['label']
    intent = 0 if intent_label == 'LABEL_0' else 1
    return intent

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6334)

# Initialize a collection in Qdrant
collection_name = "emergencies"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance="Cosine"),
)

# Load emergency data from JSON file with error handling
try:
    with open('emergency_data.json', 'r') as f:
        emergency_data = json.load(f).get('emergencies', [])
        # Insert emergency data into Qdrant
        for idx, emergency in enumerate(emergency_data):
            keywords_text = " ".join(emergency.get('keywords', []))
            embedding = embedding_model.encode(keywords_text)
            point = PointStruct(id=idx, vector=embedding, payload={"name": emergency['name'], "steps": emergency['steps']})
            client.upsert(collection_name=collection_name, points=[point])
except FileNotFoundError:
    print("Error: 'emergency_data.json' file not found.")
    emergency_data = []
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from 'emergency_data.json'.")
    emergency_data = []

# Track conversation state
conversation_state = {}

def find_best_match(user_input):
    user_embedding = embedding_model.encode(user_input)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=user_embedding,
        limit=1  # We only want the top match
    )
    
    if search_result:
        best_match = search_result[0].payload
        similarity_score = search_result[0].score
        return best_match, similarity_score
    else:
        return None, 0

def delayed_response(session_id):
    try:
        time.sleep(15)  # Simulating delay
        state = conversation_state.get(session_id)
        best_match = state.get('best_match')
        print('Yes 1')
        socketio.emit('bot_response', {"message": f"I found a match: {best_match['name']}. {best_match['steps']} Dr. Adrin is on the way."}, room=session_id)
        socketio.emit('bot_response', {"message": "Don't worry, please follow these steps, Dr. Adrin will be with you shortly."}, room=session_id)
    except Exception as e:
        print("Error: ", str(e))


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
            state['step'] = 5
        
        else:
            repeat_last_question()
            state['step'] = 1

    elif state['step'] == 3:
        if emergency_data:
            best_match, similarity_score = find_best_match(user_input)

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
        state['step'] = 1

        # Start the delayed response in a new thread
        socketio.start_background_task(delayed_response, session_id=session_id)

    
    elif state['step'] == 5:
        emit('bot_response', {"message": "Thanks for the message, I will forward it to Dr. Adrin."})
        del conversation_state[session_id]  # Reset conversation state for the session


if __name__ == "__main__":
    socketio.run(app, debug=True)
