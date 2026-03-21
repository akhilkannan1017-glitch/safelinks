from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
from dotenv import load_dotenv
load_dotenv()
import urllib.parse
import tldextract
import whois
from datetime import datetime
import math
import joblib
import numpy as np
import os
import json
import google.generativeai as genai
from threat_feed import (check_threat_db, get_feed_stats,
                          start_feed_scheduler, init_threat_db)

from database import init_db, save_scan, get_history, get_stats, clear_history
init_db()
init_threat_db()
start_feed_scheduler()

app = Flask(__name__)
CORS(app)

# ─── LOAD ML MODEL ────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phishing_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
INFO_PATH = os.path.join(os.path.dirname(__file__), 'model_info.json')

ml_model = None
ml_scaler = None
ml_info = None

try:
    ml_model = joblib.load(MODEL_PATH)
    ml_scaler = joblib.load(SCALER_PATH)
    with open(INFO_PATH) as f:
        ml_info = json.load(f)
    print(f"✓ ML Model loaded! Accuracy: {ml_info['accuracy']*100:.2f}%")
except Exception as e:
    print(f"⚠ ML Model not found, using heuristics only: {e}")


# ─── FEATURE EXTRACTION ───────────────────────────────────────
def extract_features_ml(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)
        domain = extracted.domain or ''
        suffix = extracted.suffix or ''
        subdomain = extracted.subdomain or ''
        url_lower = url.lower()

        features = []
        features.append(len(url))
        features.append(len(domain))
        features.append(len(parsed.path))
        features.append(len(subdomain))
        features.append(url.count('.'))
        features.append(url.count('-'))
        features.append(url.count('_'))
        features.append(url.count('/'))
        features.append(url.count('@'))
        features.append(url.count('?'))
        features.append(url.count('='))
        features.append(url.count('&'))
        features.append(url.count('%'))
        features.append(url.count('#'))
        features.append(sum(c.isdigit() for c in url))
        features.append(len(re.findall(r'[^a-zA-Z0-9]', url)))
        features.append(1 if parsed.scheme == 'https' else 0)
        features.append(1 if parsed.scheme == 'http' else 0)
        features.append(len(subdomain.split('.')) if subdomain else 0)
        features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0)
        features.append(sum(c.isdigit() for c in domain) / max(len(domain), 1))
        features.append(sum(c.isdigit() for c in url) / max(len(url), 1))

        phishing_words = ['login','signin','verify','update','secure','account',
                         'banking','confirm','password','paypal','ebay','amazon',
                         'apple','google','microsoft','support','helpdesk',
                         'webscr','cmd','dispatch','notification','alert',
                         'suspend','unusual','limited','restore','validate']
        features.append(sum(1 for w in phishing_words if w in url_lower))

        def entropy(s):
            if not s: return 0
            prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
            return -sum(p * math.log(p) / math.log(2.0) for p in prob)

        features.append(entropy(domain))
        features.append(entropy(url))
        features.append(entropy(parsed.path))

        suspicious_tlds = ['tk','ml','ga','cf','gq','xyz','top','pw','click',
                          'link','work','party','gdn','stream','download',
                          'bid','loan','review','win']
        features.append(1 if suffix in suspicious_tlds else 0)
        features.append(len(suffix))

        shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly',
                     'is.gd','buff.ly','adf.ly','tiny.cc','rb.gy']
        features.append(1 if any(s in url_lower for s in shorteners) else 0)

        brands = ['paypal','amazon','google','microsoft','apple',
                 'facebook','instagram','twitter','netflix','linkedin',
                 'ebay','yahoo','wellsfargo','chase','bankofamerica']
        features.append(1 if any(b in subdomain.lower() for b in brands) else 0)
        features.append(1 if any(b in parsed.path.lower() for b in brands) else 0)
        features.append(sum(1 for b in brands if b in url_lower))
        features.append(1 if '//' in url[7:] else 0)
        features.append(1 if '@' in parsed.netloc else 0)
        features.append(len([x for x in parsed.path.split('/') if x]))
        features.append(1 if parsed.port else 0)
        features.append(1 if 'https' in domain.lower() else 0)
        features.append(sum(c.isalpha() for c in url) / max(len(url), 1))
        features.append(sum(c.isdigit() for c in url) / max(len(url), 1))

        return np.array(features).reshape(1, -1)
    except:
        return None


# ─── HEURISTIC SCORING (backup) ───────────────────────────────
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['domain_length'] = len(urllib.parse.urlparse(url).netloc)
    features['path_length'] = len(urllib.parse.urlparse(url).path)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_at'] = url.count('@')
    features['num_question'] = url.count('?')
    features['num_equal'] = url.count('=')
    features['num_ampersand'] = url.count('&')
    features['num_percent'] = url.count('%')
    features['num_hash'] = url.count('#')
    features['num_digits'] = sum(c.isdigit() for c in url)
    parsed = urllib.parse.urlparse(url)
    features['has_https'] = 1 if parsed.scheme == 'https' else 0
    features['has_http'] = 1 if parsed.scheme == 'http' else 0
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain
    features['subdomain_length'] = len(subdomain)
    features['num_subdomains'] = len(subdomain.split('.')) if subdomain else 0
    features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
    suspicious_words = ['login','signin','verify','update','secure','account',
                       'banking','confirm','password','paypal','ebay','amazon',
                       'apple','google','microsoft','support','helpdesk',
                       'webscr','cmd','dispatch','notification','alert']
    url_lower = url.lower()
    features['suspicious_word_count'] = sum(1 for w in suspicious_words if w in url_lower)
    def shannon_entropy(s):
        if not s: return 0
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        return -sum(p * math.log(p) / math.log(2.0) for p in prob)
    features['domain_entropy'] = shannon_entropy(domain)
    features['url_entropy'] = shannon_entropy(url)
    suspicious_tlds = ['tk','ml','ga','cf','gq','xyz','top','pw','click',
                      'link','work','party','gdn','stream','download']
    features['suspicious_tld'] = 1 if suffix in suspicious_tlds else 0
    shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd',
                 'buff.ly','adf.ly','tiny.cc']
    features['is_shortener'] = 1 if any(s in url_lower for s in shorteners) else 0
    features['digit_ratio_domain'] = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    features['has_double_slash'] = 1 if '//' in url[7:] else 0
    brands = ['paypal','amazon','google','microsoft','apple','facebook',
             'instagram','twitter','netflix','linkedin']
    features['brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in brands) else 0
    features['brand_in_path'] = 1 if any(b in parsed.path.lower() for b in brands) else 0
    return list(features.values()), list(features.keys())


def heuristic_score(url):
    values, keys = extract_features(url)
    feat = dict(zip(keys, values))
    score = 0
    flags = []
    if feat['has_ip']: score += 40; flags.append("IP address used instead of domain name")
    if feat['has_double_slash']: score += 20; flags.append("Suspicious double-slash in URL path")
    if feat['suspicious_tld']: score += 25; flags.append("Suspicious top-level domain")
    if feat['is_shortener']: score += 15; flags.append("URL shortener detected")
    if feat['num_at'] > 0: score += 30; flags.append("@ symbol found in URL")
    if feat['brand_in_subdomain']: score += 35; flags.append("Brand name used in subdomain (impersonation)")
    if feat['brand_in_path']: score += 10; flags.append("Brand name used in path")
    if feat['suspicious_word_count'] >= 3: score += 20; flags.append(f"{feat['suspicious_word_count']} suspicious keywords found")
    elif feat['suspicious_word_count'] >= 1: score += 8
    if feat['domain_entropy'] > 3.8: score += 15; flags.append("Domain name appears randomly generated")
    if feat['url_length'] > 100: score += 10; flags.append("Unusually long URL")
    if feat['num_subdomains'] > 2: score += 15; flags.append("Excessive subdomains")
    if feat['digit_ratio_domain'] > 0.4: score += 12; flags.append("Domain contains too many digits")
    if not feat['has_https']: score += 10; flags.append("No HTTPS encryption")
    if feat['num_hyphens'] > 3: score += 8; flags.append("Multiple hyphens in URL")
    return min(score, 100), flags


# ─── COMBINED SCORING (ML + Heuristics) ───────────────────────
def combined_score(url):
    heuristic, flags = heuristic_score(url)

    # Check threat intelligence database
    is_known_threat, threat_source, threat_type = check_threat_db(url)
    if is_known_threat:
        source_names = {
            'openphish': 'OpenPhish',
            'urlhaus': 'URLhaus',
            'phishtank': 'PhishTank'
        }
        type_names = {
            'phishing': 'phishing site',
            'malware': 'malware distribution'
        }
        src = source_names.get(threat_source, threat_source)
        typ = type_names.get(threat_type, threat_type)
        flags.insert(0, f"🚨 Known {typ} — listed in {src} threat database")
        heuristic = min(heuristic + 60, 100)
    ml_score = None
    ml_confidence = None
    detection_method = "heuristic"

    if ml_model is not None:
        features = extract_features_ml(url)
        if features is not None:
            try:
                features_scaled = ml_scaler.transform(features)
                ml_pred = ml_model.predict(features_scaled)[0]
                ml_proba = ml_model.predict_proba(features_scaled)[0]
                ml_confidence = float(ml_proba[1])  # probability of phishing
                ml_score = int(ml_confidence * 100)
                detection_method = "ml+heuristic"

                # Weighted combination: 60% ML + 40% heuristic
                final_score = int(0.6 * ml_score + 0.4 * heuristic)

                # If ML is very confident, trust it more
                if ml_confidence > 0.85:
                    final_score = int(0.8 * ml_score + 0.2 * heuristic)
                    flags.append(f"🤖 ML Model: {ml_confidence*100:.0f}% phishing confidence")
                elif ml_confidence > 0.6:
                    flags.append(f"🤖 ML Model: {ml_confidence*100:.0f}% phishing probability")

                return min(final_score, 100), flags, detection_method, ml_confidence
            except Exception as e:
                print(f"ML prediction error: {e}")

    return heuristic, flags, detection_method, None


# ─── DOMAIN AGE ───────────────────────────────────────────────
def get_domain_age(url):
    try:
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list): creation = creation[0]
        if creation:
            return (datetime.now() - creation).days
    except:
        pass
    return None


# ─── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/warning')
def warning():
    return render_template('warning.html')


@app.route('/api/check', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        score, flags, method, ml_conf = combined_score(url)
        age_days = get_domain_age(url)

        if age_days is not None:
            if age_days < 30:
                score = min(score + 25, 100)
                flags.append(f"Domain is only {age_days} days old — very new!")
            elif age_days < 180:
                score = min(score + 10, 100)
                flags.append(f"Domain is only {age_days} days old")

        if score >= 60:
            verdict = "DANGEROUS"
        elif score >= 30:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}"

        save_scan(url, domain, score, verdict, 'full',
                  url.startswith('https://'), age_days,
                  flags, method, ml_conf)

        return jsonify({
            'url': url,
            'domain': domain,
            'score': score,
            'verdict': verdict,
            'flags': flags,
            'domain_age_days': age_days,
            'scan_mode': 'full',
            'detection_method': method,
            'ml_confidence': ml_conf,
            'https': url.startswith('https://')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/quick', methods=['POST'])
def quick_scan():
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url: return jsonify({'error': 'No URL provided'}), 400
    if not url.startswith(('http://', 'https://')): url = 'https://' + url

    score, flags, method, ml_conf = combined_score(url)
    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"

    flags.append("Quick scan — domain age not checked (use Full Scan for complete analysis)")

    if score >= 60: verdict = "DANGEROUS"
    elif score >= 30: verdict = "SUSPICIOUS"
    else: verdict = "SAFE"
    save_scan(url, domain, score, verdict, 'quick',
              url.startswith('https://'), None,
              flags, method, ml_conf)

    return jsonify({
        'url': url, 'domain': domain, 'score': score,
        'verdict': verdict, 'flags': flags,
        'domain_age_days': None,
        'scan_mode': 'quick',
        'detection_method': method,
        'ml_confidence': ml_conf,
        'https': url.startswith('https://')
    })


@app.route('/api/intercept', methods=['POST'])
def intercept():
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url: return jsonify({'safe': True})
    if not url.startswith(('http://', 'https://')): url = 'https://' + url
    score, flags, method, ml_conf = combined_score(url)
    return jsonify({
        'url': url, 'score': score,
        'safe': score < 60,
        'verdict': 'SAFE' if score < 60 else 'DANGEROUS',
        'flags': flags
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    if ml_info:
        return jsonify({
            'loaded': True,
            'accuracy': f"{ml_info['accuracy']*100:.2f}%",
            'training_samples': ml_info['samples'],
            'phishing_samples': ml_info['phishing'],
            'safe_samples': ml_info['safe']
        })
    return jsonify({'loaded': False, 'message': 'Using heuristic mode only'})


from flask import send_from_directory
import os as _os

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(_os.path.dirname(_os.path.abspath(__file__)), filename)


# ─── HISTORY ROUTES ───────────────────────────────────────────
@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def api_history():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(get_history(limit))

@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(get_stats())

@app.route('/api/history/clear', methods=['POST'])
def api_clear():
    clear_history()
    return jsonify({'success': True})

@app.route('/api/feed-stats', methods=['GET'])
def feed_stats():
    return jsonify(get_feed_stats())

@app.route('/chat')
def chat_page():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        req_data = request.get_json()
        user_message = req_data.get('message', '').strip()
        history = req_data.get('history', [])

        if not user_message:
            return jsonify({'error': 'No message'}), 400

        api_key = os.environ.get('GEMINI_API_KEY', '')
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-1.5-flash')

        system_prompt = """You are SafeLinks AI, a friendly cybersecurity assistant. 
Your expertise: phishing attacks, URL analysis, safe browsing, malware, social engineering.
Be friendly, clear and educational. Keep responses concise."""

        response = client.generate_content(system_prompt + "\nUser: " + user_message)
        reply = response.text
        return jsonify({'reply': reply})

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'reply': 'I am having trouble connecting right now. Please try again in a moment! 🙏'})
if __name__ == '__main__':
    app.run(debug=True, port=5000)