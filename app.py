import os
import sqlite3
import base64
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

DB_PATH = "db/faces.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def image_to_base64(image_path):
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{data}"
    except:
        pass
    return None

def get_visit_data():
    try:
        conn = get_db()
        faces = conn.execute('''
            SELECT face_id, first_seen 
            FROM faces 
            ORDER BY first_seen ASC
        ''').fetchall()
        
        visits = []
        for face in faces:
            face_id = face['face_id']
            
            entry = conn.execute('''
                SELECT timestamp, image_path 
                FROM events
                WHERE face_id = ? 
                AND event_type = "entry"
                ORDER BY event_id ASC 
                LIMIT 1
            ''', (face_id,)).fetchone()
            
            exit_event = conn.execute('''
                SELECT timestamp 
                FROM events
                WHERE face_id = ? 
                AND event_type = "exit"
                ORDER BY event_id DESC 
                LIMIT 1
            ''', (face_id,)).fetchone()
            
            entry_time = entry['timestamp'] if entry else face['first_seen']
            entry_image = image_to_base64(
                entry['image_path']) if entry else None
            exit_time = exit_event['timestamp'] if exit_event else None
            
            visits.append({
                'face_id': face_id,
                'entry_time': entry_time,
                'entry_image': entry_image,
                'exit_time': exit_time,
                'status': 'Exited' if exit_time else 'Inside'
            })
        
        conn.close()
        return visits
    except Exception as e:
        print(f"Error getting visit data: {e}")
        return []

def get_stats():
    try:
        conn = get_db()
        total = conn.execute(
            'SELECT COUNT(*) FROM faces').fetchone()[0]
        exited = conn.execute('''
            SELECT COUNT(DISTINCT face_id) FROM events
            WHERE event_type = "exit"
        ''').fetchone()[0]
        inside = total - exited
        conn.close()
        return {
            'total': total,
            'inside': inside,
            'exited': exited
        }
    except:
        return {'total': 0, 'inside': 0, 'exited': 0}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/visits')
def api_visits():
    visits = get_visit_data()
    stats = get_stats()
    return jsonify({
        'visits': visits,
        'stats': stats
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Tracker Live Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: Arial, sans-serif; 
            background: #0f0f1a; 
            color: #ffffff;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 20px 30px;
            background: #1a1a2e;
            border-radius: 12px;
            border: 1px solid #2a2a4e;
        }
        .header h1 { font-size: 24px; color: #00ff88; }
        .live-badge {
            background: #ff0000;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .stats {
            display: flex; gap: 16px; margin-bottom: 20px;
        }
        .stat-card {
            flex: 1;
            background: #1a1a2e;
            border: 1px solid #2a2a4e;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .stat-number { font-size: 42px; font-weight: bold; }
        .stat-label { font-size: 13px; color: #888; margin-top: 4px; }
        .total { color: #00ff88; }
        .inside { color: #00aaff; }
        .exited { color: #888888; }
        .table-container {
            background: #1a1a2e;
            border-radius: 12px;
            border: 1px solid #2a2a4e;
            overflow: hidden;
        }
        table { width: 100%; border-collapse: collapse; }
        thead { background: #0f0f2e; }
        thead th {
            padding: 14px 16px;
            text-align: center;
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        tbody tr {
            border-top: 1px solid #2a2a4e;
            transition: background 0.2s;
        }
        tbody tr:hover { background: #2a2a4e; }
        tbody tr.new-row {
            animation: highlight 2s ease-out;
        }
        @keyframes highlight {
            0% { background: rgba(0, 255, 136, 0.2); }
            100% { background: transparent; }
        }
        td {
            padding: 12px 16px;
            text-align: center;
            font-size: 13px;
        }
        .face-id {
            background: #007bff;
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
        }
        .face-photo {
            width: 70px;
            height: 70px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #2a2a4e;
        }
        .no-photo {
            width: 70px;
            height: 70px;
            background: #2a2a4e;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: #555;
            margin: 0 auto;
        }
        .status-inside {
            background: #00aaff22;
            color: #00aaff;
            border: 1px solid #00aaff44;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .status-exited {
            background: #88888822;
            color: #888;
            border: 1px solid #88888844;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .refresh-info {
            text-align: center; margin-top: 16px; font-size: 12px; color: #555;
        }
        .last-updated { color: #00ff88; }
        .footer { text-align: center; margin-top: 20px; font-size: 11px; color: #333; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Face Tracker Live Dashboard</h1>
            <p style="color:#888;font-size:13px;margin-top:4px;">
                Real time visitor tracking — YOLOv8 + InsightFace
            </p>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <span class="live-badge">LIVE</span>
            <span style="font-size:12px;color:#888;" id="last-updated">Connecting...</span>
        </div>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number total" id="total">0</div>
            <div class="stat-label">Total Unique Visitors</div>
        </div>
        <div class="stat-card">
            <div class="stat-number inside" id="inside">0</div>
            <div class="stat-label">Currently Inside</div>
        </div>
        <div class="stat-card">
            <div class="stat-number exited" id="exited">0</div>
            <div class="stat-label">Exited</div>
        </div>
    </div>
    
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Face ID</th>
                    <th>Photo</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="visits-table">
                <tr>
                    <td colspan="6" style="color:#555;padding:40px;">
                        Waiting for detections...
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="refresh-info">
        Auto-refreshing every 3 seconds | Last updated: <span class="last-updated" id="update-time">-</span>
    </div>
    
    <div class="footer">
        This project is a part of a hackathon run by https://katomaran.com
    </div>

<script>
    let previousCount = 0;
    
    function renderPhoto(base64) {
        if (base64) {
            return '<img src="' + base64 + '" class="face-photo">';
        }
        return '<div class="no-photo">No photo</div>';
    }
    
    function renderStatus(status) {
        if (status === 'Inside') {
            return '<span class="status-inside">Inside</span>';
        }
        return '<span class="status-exited">Exited</span>';
    }
    
    function updateDashboard() {
        fetch('/api/visits')
            .then(r => r.json())
            .then(data => {
                document.getElementById('total').textContent = data.stats.total;
                document.getElementById('inside').textContent = data.stats.inside;
                document.getElementById('exited').textContent = data.stats.exited;
                
                const tbody = document.getElementById('visits-table');
                const currentCount = data.visits.length;
                
                if (currentCount === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="color:#555;padding:40px;">Waiting for detections...</td></tr>';
                    previousCount = 0;
                    return;
                }
                
                let html = '';
                data.visits.forEach((visit, index) => {
                    const isNew = index >= previousCount;
                    const rowClass = isNew ? 'new-row' : '';
                    
                    html += '<tr class="' + rowClass + '">';
                    html += '<td style="color:#555;">' + (index + 1) + '</td>';
                    html += '<td><span class="face-id">' + visit.face_id + '</span></td>';
                    html += '<td>' + renderPhoto(visit.entry_image) + '</td>';
                    html += '<td>' + (visit.entry_time || '-') + '</td>';
                    html += '<td>' + (visit.exit_time || '-') + '</td>';
                    html += '<td>' + renderStatus(visit.status) + '</td>';
                    html += '</tr>';
                });
                
                tbody.innerHTML = html;
                previousCount = currentCount;
                
                const now = new Date();
                document.getElementById('update-time').textContent = now.toLocaleTimeString();
            })
            .catch(err => {
                console.log('Refresh error:', err);
            });
    }
    
    updateDashboard();
    setInterval(updateDashboard, 3000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Face Tracker Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
