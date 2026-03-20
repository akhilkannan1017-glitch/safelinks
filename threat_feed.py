import sqlite3
import requests
import os
import time
import threading
from datetime import datetime
import tldextract

DB_PATH = os.path.join(os.path.dirname(__file__), 'safelinks.db')

def init_threat_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS threat_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            domain TEXT,
            source TEXT,
            threat_type TEXT,
            added_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS feed_stats (
            id INTEGER PRIMARY KEY,
            last_updated TEXT,
            total_threats INTEGER DEFAULT 0,
            phishtank_count INTEGER DEFAULT 0,
            openphish_count INTEGER DEFAULT 0,
            urlhaus_count INTEGER DEFAULT 0
        )
    ''')
    c.execute('SELECT COUNT(*) FROM feed_stats')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO feed_stats VALUES (1,?,0,0,0,0)',
                  (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

def add_threat_url(url, source, threat_type='phishing'):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        c.execute('''
            INSERT OR IGNORE INTO threat_urls
            (url, domain, source, threat_type, added_at)
            VALUES (?,?,?,?,?)
        ''', (url.strip(), domain, source, threat_type,
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def fetch_openphish():
    try:
        print("  [OpenPhish] Fetching...")
        resp = requests.get(
            'https://openphish.com/feed.txt',
            timeout=15,
            headers={'User-Agent': 'SafeLinks-ThreatFeed/1.0'}
        )
        if resp.status_code == 200:
            urls = [u.strip() for u in resp.text.split('\n')
                    if u.strip().startswith('http')]
            count = 0
            for url in urls[:500]:
                if add_threat_url(url, 'openphish', 'phishing'):
                    count += 1
            print(f"  [OpenPhish] Added {count} threats")
            return count
    except Exception as e:
        print(f"  [OpenPhish] Error: {e}")
    return 0

def fetch_urlhaus():
    try:
        print("  [URLhaus] Fetching...")
        resp = requests.get(
            'https://urlhaus.abuse.ch/downloads/text_recent/',
            timeout=15,
            headers={'User-Agent': 'SafeLinks-ThreatFeed/1.0'}
        )
        if resp.status_code == 200:
            urls = [u.strip() for u in resp.text.split('\n')
                    if u.strip().startswith('http')
                    and not u.strip().startswith('#')]
            count = 0
            for url in urls[:500]:
                if add_threat_url(url, 'urlhaus', 'malware'):
                    count += 1
            print(f"  [URLhaus] Added {count} threats")
            return count
    except Exception as e:
        print(f"  [URLhaus] Error: {e}")
    return 0

def fetch_phishtank():
    try:
        print("  [PhishTank] Fetching...")
        resp = requests.get(
            'http://data.phishtank.com/data/online-valid.csv',
            timeout=30,
            headers={'User-Agent': 'SafeLinks-ThreatFeed/1.0'}
        )
        if resp.status_code == 200:
            lines = resp.text.split('\n')[1:]
            count = 0
            for line in lines[:500]:
                parts = line.split(',')
                if len(parts) > 1:
                    url = parts[1].strip().strip('"')
                    if url.startswith('http'):
                        if add_threat_url(url, 'phishtank', 'phishing'):
                            count += 1
            print(f"  [PhishTank] Added {count} threats")
            return count
    except Exception as e:
        print(f"  [PhishTank] Error: {e}")
    return 0

def update_feed_stats(op_count, uh_count, pt_count):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM threat_urls')
        total = c.fetchone()[0]
        c.execute('''
            UPDATE feed_stats SET
            last_updated=?,
            total_threats=?,
            openphish_count=openphish_count+?,
            urlhaus_count=urlhaus_count+?,
            phishtank_count=phishtank_count+?
            WHERE id=1
        ''', (datetime.now().isoformat(), total,
              op_count, uh_count, pt_count))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [Stats] Error: {e}")

def refresh_feeds():
    print(f"\n{'='*50}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing threat feeds...")
    op = fetch_openphish()
    uh = fetch_urlhaus()
    pt = fetch_phishtank()
    update_feed_stats(op, uh, pt)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM threat_urls')
    total = c.fetchone()[0]
    conn.close()
    print(f"[Feed] Total threats in database: {total}")
    print(f"{'='*50}\n")
    return total

def check_threat_db(url):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Check exact URL
        c.execute('SELECT source, threat_type FROM threat_urls WHERE url=?',
                  (url.strip(),))
        row = c.fetchone()
        if row:
            conn.close()
            return True, row[0], row[1]
        # Check domain
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        c.execute('''
            SELECT source, threat_type FROM threat_urls
            WHERE domain=? LIMIT 1
        ''', (domain,))
        row = c.fetchone()
        conn.close()
        if row:
            return True, row[0], row[1]
        return False, None, None
    except:
        return False, None, None

def get_feed_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM feed_stats WHERE id=1')
        row = c.fetchone()
        c.execute('SELECT COUNT(*) FROM threat_urls')
        total = c.fetchone()[0]
        conn.close()
        if row:
            return {
                'last_updated': row[1],
                'total_threats': total,
                'openphish_count': row[4],
                'urlhaus_count': row[5],
                'phishtank_count': row[3]
            }
    except:
        pass
    return {}

def start_feed_scheduler():
    def scheduler():
        # First run immediately
        refresh_feeds()
        # Then every hour
        while True:
            time.sleep(3600)
            refresh_feeds()
    thread = threading.Thread(target=scheduler, daemon=True)
    thread.start()
    print("✓ Threat feed scheduler started (updates every hour)")

# Initialize on import
init_threat_db()