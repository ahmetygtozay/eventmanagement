from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import logging
import psycopg2
import bcrypt
from personalized_search import recommend_events, build_rag_prompt
from feed_prompt_into_llm import query_huggingface_llm, extract_llm_answer
# ------------------ Setup ------------------ #
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection (adjust your credentials if needed)
conn = psycopg2.connect(
    dbname='eventdb',
    user='postgres',
    password='Ahmet1212.',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

# Hugging Face LLM API settings
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HUGGINGFACE_API_TOKEN = "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"

# ------------------ Data Models ------------------ #
class RecommendRequest(BaseModel):
    user_id: int
    query: str

class UserRegister(BaseModel):
    name: str
    email: str
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# ------------------ LLM Query ------------------ #
def query_llm(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        logger.error("LLM request failed: %s", response.text)
        raise HTTPException(status_code=500, detail="LLM query failed")

# ------------------ API Endpoints ------------------ #

@app.get("/")
def root():
    return {"message": "Event Recommendation API is running."}
@app.post("/api/register")
def register_user(user: UserRegister):
    try:
        hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

        cur.execute("""
            INSERT INTO users (name, email, username, password, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id
        """, (user.name, user.email, user.username, hashed_pw.decode('utf-8')))
        
        conn.commit()
        user_id = cur.fetchone()[0]
        return {"message": "User registered successfully", "user_id": user_id}
    
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=409, detail="Username or email already exists")

    except Exception as e:
        logger.exception("Registration failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
def login_user(user: UserLogin):
    try:
        cur.execute("SELECT id, password FROM users WHERE username = %s", (user.username,))
        result = cur.fetchone()

        if not result:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        user_id, hashed_pw = result

        if bcrypt.checkpw(user.password.encode('utf-8'), hashed_pw.encode('utf-8')):
            return {"message": "Login successful", "user_id": user_id}
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")

    except Exception as e:
        logger.exception("Login failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    user_id = request.user_id
    query = request.query

    # Step 1: Get retrieved events with similarity
    retrieved_events = recommend_events(user_id, query)

    # Step 2: Build prompt and get LLM response
    prompt = build_rag_prompt(query, retrieved_events)
    llm_output = query_huggingface_llm(prompt)
    final_output = extract_llm_answer(llm_output)

    # Step 3: Save recommendations to database
    try:
        for event in retrieved_events:
            event_id = event["event_id"]

            # Avoid duplicates (optional)
            cur.execute("""
                SELECT 1 FROM user_recommendations
                WHERE user_id = %s AND event_id = %s
            """, (user_id, event_id))

            if not cur.fetchone():
                cur.execute("""
                    INSERT INTO user_recommendations (user_id, event_id)
                    VALUES (%s, %s)
                """, (user_id, event_id))

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("Failed to save recommendations")

    # Step 4: Return response
    return {
        "events": retrieved_events,
        "llm_response": final_output
    }




@app.get("/api/events/{event_id}")
def get_event(event_id: int):
    try:
        cur.execute("SELECT id, title, description, category, tags, location FROM events WHERE id = %s", (event_id,))
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Event not found")

        event = {
            "id": result[0],
            "title": result[1],
            "description": result[2],
            "category": result[3],
            "tags": result[4],
            "location": result[5]
        }
        return event

    except Exception as e:
        logger.exception("Error fetching event")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: int):
    try:
        cur.execute("""
            SELECT e.id, e.title, e.category, e.location
            FROM user_event_history ueh
            JOIN events e ON ueh.event_id = e.id
            WHERE ueh.user_id = %s
            ORDER BY ueh.interaction_time DESC
            LIMIT 10
        """, (user_id,))
        rows = cur.fetchall()

        history = [
            {
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "location": row[3]
            } for row in rows
        ]

        return {"user_id": user_id, "history": history}

    except Exception as e:
        logger.exception("Error fetching user history")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/recommendations")
def get_user_recommendations(user_id: int):
    try:
        cur.execute("""
            SELECT e.id, e.title, e.description, e.category, e.tags, e.location, ur.recommended_at
            FROM user_recommendations ur
            JOIN events e ON ur.event_id = e.id
            WHERE ur.user_id = %s
            ORDER BY ur.recommended_at DESC
        """, (user_id,))
        rows = cur.fetchall()

        recommendations = [
            {
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "tags": row[4],
                "location": row[5],
                "recommended_at": row[6]
            } for row in rows
        ]

        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        logger.exception("Error fetching saved recommendations")
        raise HTTPException(status_code=500, detail=str(e))
