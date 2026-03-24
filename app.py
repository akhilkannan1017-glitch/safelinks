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


# ─── GLOBAL WHITELIST ─────────────────────────────────────────
TRUSTED_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
    'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
    'github.com', 'stackoverflow.com', 'reddit.com', 'netflix.com',
    'microsoft.com', 'apple.com', 'yahoo.com', 'bing.com',
    'zoom.us', 'dropbox.com', 'spotify.com', 'twitch.tv',
    'ebay.com', 'paypal.com', 'wordpress.com', 'medium.com',
    'bbc.com', 'cnn.com', 'nytimes.com', 'techcrunch.com',
    'mozilla.org', 'python.org', 'npmjs.com', 'docker.com',
    'heroku.com', 'digitalocean.com', 'cloudflare.com',
    'stripe.com', 'shopify.com', 'salesforce.com', 'slack.com',
    'adobe.com', 'oracle.com', 'ibm.com', 'cisco.com',
    'intel.com', 'nvidia.com', 'samsung.com', 'hp.com',
    'dell.com', 'lenovo.com', 'asus.com', 'flipkart.com',
    'paytm.com', 'phonepe.com', 'irctc.co.in', 'booking.com',
    'airbnb.com', 'coursera.org', 'udemy.com', 'khanacademy.org',
    'w3schools.com', 'gitlab.com', 'bitbucket.org', 'notion.so',
    'figma.com', 'canva.com', 'trello.com', 'asana.com',
    'whatsapp.com', 'telegram.org', 'discord.com', 'signal.org',
    'onrender.com', 'netlify.app', 'vercel.app', 'pages.dev',
    'githubusercontent.com', 'githubassets.com', 'githubapp.com'
]

def is_trusted_domain(url):
    try:
        extracted = tldextract.extract(url)
        full_domain = f"{extracted.domain}.{extracted.suffix}".lower()
        for trusted in TRUSTED_DOMAINS:
            if full_domain == trusted or full_domain.endswith('.' + trusted):
                return True
    except:
        pass
    return False


# ─── FEATURE EXTRACTION ───────────────────────────────────────
def extract_features(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)
        domain = extracted.domain or ''
        suffix = extracted.suffix or ''
        subdomain = extracted.subdomain or ''

        features = {}
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['path_length'] = len(parsed.path)
        features['subdomain_length'] = len(subdomain)
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
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_http'] = 1 if parsed.scheme == 'http' else 0
        features['num_subdomains'] = len(subdomain.split('.')) if subdomain else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0

        def shannon_entropy(s):
            if not s: return 0
            prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
            return -sum(p * math.log(p) / math.log(2.0) for p in prob)

        features['domain_entropy'] = shannon_entropy(domain)
        features['url_entropy'] = shannon_entropy(url)

        suspicious_tlds = ['tk','ml','ga','cf','gq','xyz','top','pw','click',
                           'link','work','party','gdn','stream','download',
                           'bid','loan','review','win']
        features['suspicious_tld'] = 1 if suffix in suspicious_tlds else 0

        shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd','buff.ly','adf.ly','tiny.cc']
        url_lower = url.lower()
        features['is_shortener'] = 1 if any(s in url_lower for s in shorteners) else 0
        features['digit_ratio_domain'] = sum(c.isdigit() for c in domain) / max(len(domain), 1)
        features['has_double_slash'] = 1 if '//' in url[7:] else 0

        brands = ['paypal','amazon','google','microsoft','apple','facebook','instagram','twitter','netflix','linkedin']
        features['brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in brands) else 0
        features['brand_in_path'] = 1 if any(b in parsed.path.lower() for b in brands) else 0

        return list(features.values()), list(features.keys())

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return [0] * 24, ['f' + str(i) for i in range(24)]


# ─── HEURISTIC SCORING ────────────────────────────────────────
def heuristic_score(url):
    # Whitelist check — already handled in combined_score but kept as safety net
    if is_trusted_domain(url):
        return 0, []

    values, keys = extract_features(url)
    feat = dict(zip(keys, values))
    score = 0
    flags = []

    url_lower = url.lower()
    parsed = urllib.parse.urlparse(url)
    extracted2 = tldextract.extract(url)
    domain = extracted2.domain
    subdomain2 = extracted2.subdomain
    suffix = extracted2.suffix

    if feat['has_ip']:
        score += 45
        flags.append("IP address used instead of domain name")

    if feat['num_at'] > 0:
        score += 40
        flags.append("@ symbol found in URL")

    if feat['has_double_slash']:
        score += 25
        flags.append("Suspicious double-slash in URL path")

    brands = ['paypal','amazon','google','microsoft','apple','facebook',
              'instagram','twitter','netflix','linkedin','ebay','yahoo',
              'wellsfargo','chase','bankofamerica','citibank','hsbc',
              'barclays','hdfc','icici','sbi']

    for brand in brands:
        if brand in subdomain2.lower():
            score += 40
            flags.append(f"Brand name '{brand}' used in subdomain (impersonation)")
            break
        if (brand in domain.lower() and domain.lower() != brand and len(domain) > len(brand) + 2):
            score += 30
            flags.append(f"Brand name '{brand}' in domain (possible impersonation)")
            break

    bad_tlds = ['tk','ml','ga','cf','gq','xyz','top','pw','click','link','work',
                'party','gdn','stream','download','bid','loan','review','win',
                'racing','cricket','science','accountant','date','faith']
    if suffix in bad_tlds:
        score += 30
        flags.append("Suspicious top-level domain")

    shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd',
                  'buff.ly','adf.ly','tiny.cc','rb.gy','cutt.ly','short.io','bl.ink']
    if any(s in url_lower for s in shorteners):
        score += 20
        flags.append("URL shortener detected")

    phish_words = ['login','signin','verify','update','secure','account','banking',
                   'confirm','password','helpdesk','webscr','cmd','dispatch',
                   'notification','alert','suspend','unusual','limited','restore',
                   'validate','authenticate','credential','authorize']
    kw_count = sum(1 for w in phish_words if w in url_lower)
    if kw_count >= 5:
        score += 35
        flags.append(f"{kw_count} suspicious keywords found")
    elif kw_count >= 3:
        score += 25
        flags.append(f"{kw_count} suspicious keywords found")
    elif kw_count >= 1:
        score += 8

    if not url.startswith('https://'):
        score += 15
        flags.append("No HTTPS encryption")

    if feat['domain_entropy'] > 4.2:
        score += 20
        flags.append("Domain name appears randomly generated")
    elif feat['domain_entropy'] > 4.0:
        score += 10

    if len(url) > 200:
        score += 15
        flags.append("Unusually long URL")
    elif len(url) > 150:
        score += 8

    if feat['num_subdomains'] > 4:
        score += 20
        flags.append("Excessive subdomains")
    elif feat['num_subdomains'] > 3:
        score += 10

    if feat['digit_ratio_domain'] > 0.6:
        score += 15
        flags.append("Domain contains too many digits")

    if feat['num_hyphens'] > 5:
        score += 12
        flags.append("Multiple hyphens in URL")

    # ── Typosquatting — now includes paypai ───────────────────
    typos = {
        'paypa1':'paypal','paypall':'paypal','paypai':'paypal',
        'arnazon':'amazon','amazom':'amazon','amaz0n':'amazon',
        'gooogle':'google','goggle':'google','g00gle':'google',
        'faceb00k':'facebook','facebok':'facebook',
        'micosoft':'microsoft','microsofft':'microsoft',
        'netfl1x':'netflix','netflx':'netflix',
        'lnkedin':'linkedin','linkedln':'linkedin'
    }
    for typo, real in typos.items():
        if typo in domain.lower():
            score += 45
            flags.append(f"Typosquatting detected — mimics '{real}'")
            break

    suspicious_paths = ['/wp-login','/admin/login','/secure/login',
                        '/account/login','/signin/confirm','/verify/account',
                        '/update/password','/confirm/account','/suspend/verify']
    path = parsed.path.lower()
    for sp in suspicious_paths:
        if sp in path:
            score += 20
            flags.append("Suspicious path pattern detected")
            break

    return min(score, 100), flags


# ─── COMBINED SCORE — CORRECT ORDER ───────────────────────────
def combined_score(url):
    # ── STEP 1: WHITELIST — runs before EVERYTHING ────────────
    if is_trusted_domain(url):
        return 0, [], "whitelist", None

    # ── STEP 2: Heuristics ────────────────────────────────────
    heuristic, flags = heuristic_score(url)

    # ── STEP 3: Threat feed (skipped for trusted domains) ─────
    is_known_threat, threat_source, threat_type = check_threat_db(url)
    if is_known_threat:
        source_names = {'openphish':'OpenPhish','urlhaus':'URLhaus','phishtank':'PhishTank'}
        type_names = {'phishing':'phishing site','malware':'malware distribution'}
        src = source_names.get(threat_source, threat_source)
        typ = type_names.get(threat_type, threat_type)
        flags.insert(0, f"🚨 Known {typ} — listed in {src} threat database")
        heuristic = min(heuristic + 60, 100)

    # ── STEP 4: ML Model (skipped for trusted domains) ────────
    ml_score = None
    ml_confidence = None
    detection_method = "heuristic"

    if ml_model is not None:
        features, _ = extract_features(url)
        if features is not None:
            try:
                features_array = np.array(features).reshape(1, -1)
                features_scaled = ml_scaler.transform(features_array)
                ml_pred = ml_model.predict(features_scaled)[0]
                ml_proba = ml_model.predict_proba(features_scaled)[0]
                ml_confidence = float(ml_proba[1])
                ml_score = int(ml_confidence * 100)
                detection_method = "ml+heuristic"

                if ml_confidence > 0.8:
                    final_score = int(0.7 * ml_score + 0.3 * heuristic)
                    flags.append(f"🤖 ML Model: {ml_confidence*100:.0f}% phishing confidence")
                elif ml_confidence < 0.2 and heuristic < 20:
                    final_score = int(0.6 * ml_score + 0.4 * heuristic)
                else:
                    final_score = max(int(0.6 * ml_score + 0.4 * heuristic), heuristic)
                    if ml_confidence > 0.5:
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

        # Skip domain age penalty for trusted domains
        if age_days is not None and not is_trusted_domain(url):
            if age_days < 30:
                score = min(score + 25, 100)
                flags.append(f"Domain is only {age_days} days old — very new!")
            elif age_days < 180:
                score = min(score + 10, 100)
                flags.append(f"Domain is only {age_days} days old")

        if score >= 55:
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
            'url': url, 'domain': domain, 'score': score,
            'verdict': verdict, 'flags': flags,
            'domain_age_days': age_days, 'scan_mode': 'full',
            'detection_method': method, 'ml_confidence': ml_conf,
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

    if not is_trusted_domain(url):
        flags.append("Quick scan — domain age not checked (use Full Scan for complete analysis)")

    if score >= 55: verdict = "DANGEROUS"
    elif score >= 30: verdict = "SUSPICIOUS"
    else: verdict = "SAFE"

    save_scan(url, domain, score, verdict, 'quick',
              url.startswith('https://'), None, flags, method, ml_conf)

    return jsonify({
        'url': url, 'domain': domain, 'score': score,
        'verdict': verdict, 'flags': flags,
        'domain_age_days': None, 'scan_mode': 'quick',
        'detection_method': method, 'ml_confidence': ml_conf,
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
        client = genai.GenerativeModel('gemini-2.0-flash')

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