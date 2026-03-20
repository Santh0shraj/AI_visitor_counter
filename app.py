import json
import sqlite3
from flask import Flask, jsonify, render_template_string

from modules.database import DatabaseManager
from modules.counter import VisitorCounter

# Initialize Flask application
app = Flask(__name__)

# 1. Load config.json to get db_path
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    db_path = config.get("db_path", "db/faces.db")
except FileNotFoundError:
    db_path = "db/faces.db"

# 2. Initialize DatabaseManager and VisitorCounter from modules
db_manager = DatabaseManager(db_path)
counter = VisitorCounter(db_manager)

def get_recent_events(limit=10):
    """
    Helper function to query the SQLite database directly 
    for the most recent events, ordered descending by ID.
    
    Returns a list of dictionaries.
    """
    # Use the protected _get_connection from DatabaseManager
    conn = db_manager._get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT face_id, event_type, timestamp, image_path 
            FROM events 
            ORDER BY event_id DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        events = []
        for row in rows:
            events.append({
                "face_id": row[0],
                "event_type": row[1],
                "timestamp": row[2],
                "image_path": row[3]
            })
        return events
    except sqlite3.Error as e:
        print(f"Failed to query recent events: {e}")
        return []
    finally:
        conn.close()

# Inline HTML template with a <meta> refresh tag (5 seconds)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Tracker Dashboard</title>
    <!-- Auto-refresh every 5 seconds -->
    <meta http-equiv="refresh" content="5">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background-color: #f9f9f9;
        }
        h1 { color: #333; }
        .stats { 
            font-size: 2.5em; 
            margin-bottom: 30px; 
            background: #eef;
            display: inline-block;
            padding: 15px 30px;
            border-radius: 8px;
            border: 1px solid #bce;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            background: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }
        th { 
            background-color: #f2f2f2; 
        }
        tr:nth-child(even) { background-color: #fafafa; }
    </style>
</head>
<body>
    <h1>Face Tracker Dashboard</h1>
    
    <div class="stats">
        <strong>Unique Visitors:</strong> {{ visitor_count }}
    </div>
    
    <h2>Recent Events</h2>
    <table>
        <tr>
            <th>Face ID</th>
            <th>Event Type</th>
            <th>Timestamp</th>
            <th>Image Path</th>
        </tr>
        {% for event in events %}
        <tr>
            <td>{{ event.face_id }}</td>
            <td style="text-transform: uppercase;"><strong>{{ event.event_type }}</strong></td>
            <td>{{ event.timestamp }}</td>
            <td>{{ event.image_path }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

@app.route('/')
def index():
    """
    GET / 
    Serves a simple functional HTML dashboard using the loaded inline template.
    Displays total unique visitors and a table of the last 10 events.
    """
    total_unique = counter.get_unique_count()
    recent = get_recent_events(limit=10)
    
    return render_template_string(HTML_TEMPLATE, visitor_count=total_unique, events=recent)

@app.route('/api/stats')
def api_stats():
    """
    GET /api/stats
    Returns raw statistical data dynamically in JSON format.
    """
    total_unique = counter.get_unique_count()
    recent = get_recent_events(limit=10)
    
    return jsonify({
        "unique_visitors": total_unique,
        "recent_events": recent
    })

if __name__ == '__main__':
    # Run the server on the default port 5000 as requested
    app.run(host='0.0.0.0', port=5000, debug=False)
