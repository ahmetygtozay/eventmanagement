import psycopg2
import random
from datetime import datetime, timedelta

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="eventdb",
    user="postgres",
    password="Ahmet1212.",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Fetch all user IDs
cur.execute("SELECT id FROM users")
user_ids = [row[0] for row in cur.fetchall()]

# Fetch all event IDs
cur.execute("SELECT id FROM events")
event_ids = [row[0] for row in cur.fetchall()]

# Clear existing history if needed (optional)
# cur.execute("DELETE FROM user_event_history")

# Insert random user-event interactions
for user_id in user_ids:
    num_events = random.randint(3, 7)
    selected_events = random.sample(event_ids, num_events)
    
    for event_id in selected_events:
        # Randomly generate a timestamp in the last 60 days
        random_days = random.randint(1, 60)
        interaction_time = datetime.now() - timedelta(days=random_days)
        
        cur.execute("""
            INSERT INTO user_event_history (user_id, event_id, interaction_time)
            VALUES (%s, %s, %s)
        """, (user_id, event_id, interaction_time))

conn.commit()
print("Random user-event interaction history inserted successfully.")

cur.close()
conn.close()
