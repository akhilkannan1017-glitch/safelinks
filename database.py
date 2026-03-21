import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'safelinks.db')

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            domain TEXT,
            score INTEGER,
            verdict TEXT,
            scan_mode TEXT,
            https INTEGER,
            domain_age_days INTEGER,
            flags TEXT,
            detection_method TEXT,
            ml_confidence REAL,
            scanned_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_scans INTEGER DEFAULT 0,
            total_safe INTEGER DEFAULT 0,
            total_suspicious INTEGER DEFAULT 0,
            total_dangerous INTEGER DEFAULT 0,
            last_updated TEXT
        )
    ''')
    c.execute('SELECT COUNT(*) FROM stats')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO stats VALUES (1,0,0,0,0,?)', (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

def save_scan(url, domain, score, verdict, scan_mode, https,
              domain_age_days, flags, detection_method, ml_confidence):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    import json
    c.execute('''
        INSERT INTO scan_history
        (url, domain, score, verdict, scan_mode, https,
         domain_age_days, flags, detection_method, ml_confidence, scanned_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        url, domain, score, verdict, scan_mode,
        1 if https else 0,
        domain_age_days,
        json.dumps(flags),
        detection_method,
        ml_confidence,
        datetime.now().isoformat()
    ))
    # Update stats
    c.execute('UPDATE stats SET total_scans = total_scans + 1, last_updated = ? WHERE id=1',
              (datetime.now().isoformat(),))
    if verdict == 'SAFE':
        c.execute('UPDATE stats SET total_safe = total_safe + 1 WHERE id=1')
    elif verdict == 'SUSPICIOUS':
        c.execute('UPDATE stats SET total_suspicious = total_suspicious + 1 WHERE id=1')
    elif verdict == 'DANGEROUS':
        c.execute('UPDATE stats SET total_dangerous = total_dangerous + 1 WHERE id=1')
    conn.commit()
    conn.close()

def get_history(limit=50):
    conn = sqlite3.connect(DB_PATH,timeout=30)
    c = conn.cursor()
    import json
    c.execute('''
        SELECT id, url, domain, score, verdict, scan_mode,
               https, domain_age_days, flags, detection_method,
               ml_confidence, scanned_at
        FROM scan_history
        ORDER BY scanned_at DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append({
            'id': row[0],
            'url': row[1],
            'domain': row[2],
            'score': row[3],
            'verdict': row[4],
            'scan_mode': row[5],
            'https': bool(row[6]),
            'domain_age_days': row[7],
            'flags': json.loads(row[8]) if row[8] else [],
            'detection_method': row[9],
            'ml_confidence': row[10],
            'scanned_at': row[11]
        })
    return result

def get_stats():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('SELECT * FROM stats WHERE id=1')
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'total_scans': row[1],
            'total_safe': row[2],
            'total_suspicious': row[3],
            'total_dangerous': row[4],
            'last_updated': row[5]
        }
    return {}

def clear_history():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute('DELETE FROM scan_history')
    c.execute('UPDATE stats SET total_scans=0, total_safe=0, total_suspicious=0, total_dangerous=0 WHERE id=1')
    conn.commit()
    conn.close()