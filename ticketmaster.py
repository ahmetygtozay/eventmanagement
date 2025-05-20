import requests
import psycopg2

API_KEY = 'H4egWTa1JpsEkY8GqDEjxEvevaJ5zbAc'
COUNTRY_CODE = 'TR'  # Change to your target country
EVENT_CLASSIFICATION = 'music'

# Connect to your PostgreSQL DB
conn = psycopg2.connect(
    dbname='eventdb',
    user='postgres',
    password='Ahmet1212.',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

def fetch_events(page=0):
    url = 'https://app.ticketmaster.com/discovery/v2/events.json'
    params = {
        'apikey': API_KEY,
        'countryCode': COUNTRY_CODE,
        'classificationName': EVENT_CLASSIFICATION,
        'size': 50,
        'page': page
    }

    response = requests.get(url, params=params)
    data = response.json()

    if '_embedded' not in data:
        print("No events found.")
        return []

    return data['_embedded']['events']

def insert_event(event):
    title = event.get('name', 'No Title')
    description = get_event_description(event)
    category = event['classifications'][0]['segment']['name'] if event.get('classifications') else 'General'

    tags = []
    if event.get('classifications') and 'genre' in event['classifications'][0]:
        genre = event['classifications'][0]['genre']
        if isinstance(genre, dict) and 'name' in genre:
            tags = [genre['name']]

    location = event['_embedded']['venues'][0]['name'] + ', ' + event['_embedded']['venues'][0]['city']['name']
    start_time = event['dates']['start']['localDate'] + ' ' + event['dates']['start'].get('localTime', '00:00:00')

    end_time = None
    if 'end' in event['dates']:
        end_time = event['dates']['end']['localDate'] + ' ' + event['dates']['end'].get('localTime', '23:59:59')

    cur.execute("""
        INSERT INTO events (title, description, category, tags, location, start_time, end_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (title, description, category, tags, location, start_time, end_time))
    conn.commit()
    
def get_event_description(event):
    info = event.get('info', '')
    please_note = event.get('pleaseNote', '')
    
    if info and please_note:
        return info + " " + please_note
    elif info:
        return info
    elif please_note:
        return please_note
    else:
        return 'No description available'

# Fetch and insert events
pages = 3  # Fetch 3 pages of events (adjust as needed)
for page in range(pages):
    events = fetch_events(page)
    for event in events:
        insert_event(event)

print("Done: Ticketmaster events inserted.")

# Close DB connection
cur.close()
conn.close()
