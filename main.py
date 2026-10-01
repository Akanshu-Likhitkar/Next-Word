from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn

# FastAPI app initialize
app = FastAPI(title="Next Word AI API")

# CORS enable karna zaroori hai taaki HTML file API ko call kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me ise apne frontend domain par set karein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 1. Load Model, Tokenizer and max_len
# =========================================================
print("Loading model and assets, please wait...")
try:
    model = load_model("BiGRU_Model.keras")
    with open("tokenizer (1).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    print("✅ Assets loaded successfully!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")

# =========================================================
# 2. Define Request Schema (Data Validation)
# =========================================================
class GenerateRequest(BaseModel):
    text: str
    word_count: int = 1

# =========================================================
# 3. Text Generation Logic
# =========================================================
def generate_text_logic(model, tokenizer, text, max_len, num_words):
    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = sequence[-max_len:]
        sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="post"
        )
        # verbose=0 terminal me extra logs hide karne ke liye
        pred = model.predict(sequence, verbose=0)
        pred_index = np.argmax(pred)
        next_word = tokenizer.index_word.get(pred_index, "")
        
        if not next_word:
            break
            
        text += " " + next_word
        
    return text

# =========================================================
# 4. API Endpoint definition
# =========================================================
@app.post("/generate")
def generate(request: GenerateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please enter some text first.")
    
    try:
        # Generate prediction
        result = generate_text_logic(model, tokenizer, request.text, max_len, request.word_count)
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    # Server start (FastAPI usually runs on port 8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)