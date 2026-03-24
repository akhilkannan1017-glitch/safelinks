import pandas as pd
import numpy as np
import re
import urllib.parse
import math
import tldextract
import joblib
import requests
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SafeLinks ML Model Trainer")
print("=" * 60)

# ─── FEATURE EXTRACTION ────────────────────────────────────────
def extract_features(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        features = {}
        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)
        domain = extracted.domain or ''
        suffix = extracted.suffix or ''
        subdomain = extracted.subdomain or ''
        url_lower = url.lower()

        # Length features
        features['url_length'] = len(url)
        features['domain_length'] = len(domain)
        features['path_length'] = len(parsed.path)
        features['subdomain_length'] = len(subdomain)

        # Count features
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
        features['num_special'] = len(re.findall(r'[^a-zA-Z0-9]', url))

        # Protocol
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_http'] = 1 if parsed.scheme == 'http' else 0

        # Domain features
        features['num_subdomains'] = len(subdomain.split('.')) if subdomain else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
        features['digit_ratio_domain'] = sum(c.isdigit() for c in domain) / max(len(domain), 1)
        features['digit_ratio_url'] = features['num_digits'] / max(len(url), 1)

        # Suspicious keywords
        phishing_words = [
            'login', 'signin', 'verify', 'update', 'secure', 'account',
            'banking', 'confirm', 'password', 'paypal', 'ebay', 'amazon',
            'apple', 'google', 'microsoft', 'support', 'helpdesk',
            'webscr', 'cmd', 'dispatch', 'notification', 'alert',
            'suspend', 'unusual', 'limited', 'restore', 'validate'
        ]
        features['suspicious_word_count'] = sum(1 for w in phishing_words if w in url_lower)

        # Entropy
        def entropy(s):
            if not s: return 0
            prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
            return -sum(p * math.log(p) / math.log(2.0) for p in prob)

        features['domain_entropy'] = entropy(domain)
        features['url_entropy'] = entropy(url)
        features['path_entropy'] = entropy(parsed.path)

        # TLD features
        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'pw',
                          'click', 'link', 'work', 'party', 'gdn', 'stream',
                          'download', 'bid', 'loan', 'review', 'win']
        features['suspicious_tld'] = 1 if suffix in suspicious_tlds else 0
        features['tld_length'] = len(suffix)

        # URL shortener
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly',
                     'is.gd', 'buff.ly', 'adf.ly', 'tiny.cc', 'rb.gy']
        features['is_shortener'] = 1 if any(s in url_lower for s in shorteners) else 0

        # Brand impersonation
        brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple',
                 'facebook', 'instagram', 'twitter', 'netflix', 'linkedin',
                 'ebay', 'yahoo', 'wellsfargo', 'chase', 'bankofamerica']
        features['brand_in_subdomain'] = 1 if any(b in subdomain.lower() for b in brands) else 0
        features['brand_in_path'] = 1 if any(b in parsed.path.lower() for b in brands) else 0
        features['brand_count'] = sum(1 for b in brands if b in url_lower)

        # Structural anomalies
        features['has_double_slash'] = 1 if '//' in url[7:] else 0
        features['has_at_symbol'] = 1 if '@' in parsed.netloc else 0
        features['url_depth'] = len([x for x in parsed.path.split('/') if x])
        features['has_port'] = 1 if parsed.port else 0
        features['is_https_in_domain'] = 1 if 'https' in domain.lower() else 0

        # Ratio features
        features['letters_ratio'] = sum(c.isalpha() for c in url) / max(len(url), 1)
        features['digits_ratio'] = features['num_digits'] / max(len(url), 1)

        return list(features.values())

    except Exception as e:
        return None

FEATURE_NAMES = [
    'url_length', 'domain_length', 'path_length', 'subdomain_length',
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_at', 'num_question', 'num_equal', 'num_ampersand',
    'num_percent', 'num_hash', 'num_digits', 'num_special',
    'has_https', 'has_http', 'num_subdomains', 'has_ip',
    'digit_ratio_domain', 'digit_ratio_url', 'suspicious_word_count',
    'domain_entropy', 'url_entropy', 'path_entropy',
    'suspicious_tld', 'tld_length', 'is_shortener',
    'brand_in_subdomain', 'brand_in_path', 'brand_count',
    'has_double_slash', 'has_at_symbol', 'url_depth',
    'has_port', 'is_https_in_domain', 'letters_ratio', 'digits_ratio'
]

# ─── DOWNLOAD REAL DATASETS ───────────────────────────────────
print("Downloading real phishing datasets...")

phishing_urls = []
safe_urls = []

# Download from OpenPhish (free, no key needed)
try:
    import urllib.request
    print("  Fetching OpenPhish...")
    req = urllib.request.Request(
        'https://openphish.com/feed.txt',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read().decode('utf-8')
    urls = [u.strip() for u in data.split('\n')
            if u.strip().startswith('http')]
    phishing_urls.extend(urls[:2000])
    print(f"  OpenPhish: {len(urls)} URLs")
except Exception as e:
    print(f"  OpenPhish failed: {e}")

# Download from URLhaus
try:
    print("  Fetching URLhaus...")
    req = urllib.request.Request(
        'https://urlhaus.abuse.ch/downloads/text_recent/',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read().decode('utf-8')
    urls = [u.strip() for u in data.split('\n')
            if u.strip().startswith('http')
            and not u.strip().startswith('#')]
    phishing_urls.extend(urls[:2000])
    print(f"  URLhaus: {len(urls)} URLs")
except Exception as e:
    print(f"  URLhaus failed: {e}")

# Download PhishTank
try:
    print("  Fetching PhishTank...")
    req = urllib.request.Request(
        'http://data.phishtank.com/data/online-valid.csv',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read().decode('utf-8', errors='ignore')
    lines = data.split('\n')[1:]
    for line in lines[:3000]:
        parts = line.split(',')
        if len(parts) > 1:
            url = parts[1].strip().strip('"')
            if url.startswith('http'):
                phishing_urls.append(url)
    print(f"  PhishTank: added URLs")
except Exception as e:
    print(f"  PhishTank failed: {e}")

# Built-in phishing patterns (always included)
builtin_phishing = [
    "http://paypal-secure-login.tk/verify/account",
    "http://amazon-account-update.ml/confirm/password",
    "http://google-account-signin-verify.xyz/update/password/confirm/secure",
    "http://apple-id-locked.cf/signin/confirm/verify",
    "http://microsoft-support-alert.ga/helpdesk/verify",
    "http://192.168.1.1/login/verify/account/password",
    "http://secure-paypal.com.login.tk/webscr",
    "http://www.paypal.com.secure.login.fakesite.com/cmd",
    "http://amazon.com.account.update.malicious.tk/confirm",
    "http://signin.google.com.phishing.ml/accounts",
    "http://apple.com.id.verify.malicious.xyz/apple-id",
    "http://microsoft.support.alert.work/helpdesk/verify",
    "http://192.0.2.1/banking/login/secure",
    "http://10.0.0.1/paypal/login",
    "http://secure-login-verify.tk/account/update",
    "http://account-verify.ml/signin/password",
    "http://login-secure.xyz/update/credentials",
    "http://verify-account.cf/confirm/banking",
    "http://update-credentials.ga/secure/login",
    "http://banking-alert.work/suspend/account",
    "http://bit.ly/phishing-link",
    "http://tinyurl.com/fake-paypal",
    "http://paypal@malicious-site.com/login",
    "http://secure.apple.com@phishing.tk/signin",
    "http://192.168.0.1//login//verify",
    "http://update.microsoft.com.fakesite.ml/security",
    "http://ebay.com.account-suspended.xyz/verify",
    "http://chase-bank-alert.tk/online/login",
    "http://wellsfargo-secure.ml/banking/signin",
    "http://bankofamerica.alert.xyz/account/verify",
    "http://netflix.com.login.tk/account/update",
    "http://instagram.com.verify.ml/account/signin",
    "http://facebook.com.security.alert.xyz/login",
    "http://linkedin.com.account.verify.tk/signin",
    "http://twitter.com.suspended.ml/verify/account",
    "http://xn--pypal-4ve.com/verify",
    "http://paypa1.com/login",
    "http://arnazon.com/account",
    "http://micosoft.com/update",
    "http://gooogle.com/signin",
    "http://faceb00k.com/login",
] + [
    f"http://random{i}phish.tk/login/verify/account/secure" for i in range(200)
] + [
    f"http://secure{i}bank.ml/account/update/password/confirm" for i in range(200)
] + [
    f"http://{i}.{i}.{i}.{i}/login/verify" for i in range(1, 100)
] + [
    f"http://paypal-login-{i}.xyz/verify/account" for i in range(100)
] + [
    f"http://amazon-update-{i}.ml/confirm/password" for i in range(100)
] + [
    f"http://apple-id-verify-{i}.tk/signin/confirm" for i in range(100)
]

phishing_urls.extend(builtin_phishing)

# Safe URLs
safe_urls = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.twitter.com",
    "https://www.instagram.com",
    "https://www.linkedin.com",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.reddit.com",
    "https://www.netflix.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.yahoo.com",
    "https://www.bing.com",
    "https://www.zoom.us",
    "https://www.dropbox.com",
    "https://www.spotify.com",
    "https://www.twitch.tv",
    "https://www.ebay.com",
    "https://www.paypal.com",
    "https://www.wordpress.com",
    "https://www.tumblr.com",
    "https://www.pinterest.com",
    "https://www.quora.com",
    "https://www.medium.com",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.nytimes.com",
    "https://www.techcrunch.com",
    "https://www.wired.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.khanacademy.org",
    "https://www.w3schools.com",
    "https://www.mozilla.org",
    "https://www.python.org",
    "https://www.npmjs.com",
    "https://www.docker.com",
    "https://www.heroku.com",
    "https://www.digitalocean.com",
    "https://www.cloudflare.com",
    "https://www.stripe.com",
    "https://www.shopify.com",
    "https://www.salesforce.com",
    "https://www.hubspot.com",
    "https://www.zendesk.com",
    "https://www.atlassian.com",
    "https://www.slack.com",
    "https://docs.google.com",
    "https://mail.google.com",
    "https://drive.google.com",
    "https://accounts.google.com/signin",
    "https://myaccount.google.com",
    "https://support.microsoft.com",
    "https://office.microsoft.com",
    "https://login.microsoftonline.com",
    "https://portal.azure.com",
    "https://docs.microsoft.com",
    "https://www.adobe.com",
    "https://www.oracle.com",
    "https://www.ibm.com",
    "https://www.cisco.com",
    "https://www.intel.com",
    "https://www.nvidia.com",
    "https://www.samsung.com",
    "https://www.sony.com",
    "https://www.lg.com",
    "https://www.hp.com",
    "https://www.dell.com",
    "https://www.lenovo.com",
    "https://www.asus.com",
    "https://www.acer.com",
    "https://www.flipkart.com",
    "https://www.myntra.com",
    "https://www.swiggy.com",
    "https://www.zomato.com",
    "https://www.paytm.com",
    "https://www.phonepe.com",
    "https://www.gpay.com",
    "https://www.irctc.co.in",
    "https://www.makemytrip.com",
    "https://www.booking.com",
    "https://www.airbnb.com",
    "https://www.tripadvisor.com",
] + [
    f"https://www.site{i}.com/page/about/contact" for i in range(300)
] + [
    f"https://blog{i}.wordpress.com/2024/article/post" for i in range(200)
] + [
    f"https://shop{i}.shopify.com/products/item" for i in range(200)
] + [
    f"https://news{i}.medium.com/article/story" for i in range(200)
]

# Remove duplicates
phishing_urls = list(set(phishing_urls))
safe_urls = list(set(safe_urls))

print(f"\nTotal phishing URLs: {len(phishing_urls)}")
print(f"Total safe URLs: {len(safe_urls)}")
# Try to download PhishTank data
print("   Attempting to download PhishTank dataset...")
try:
    import urllib.request
    url_pt = "http://data.phishtank.com/data/online-valid.csv"
    urllib.request.urlretrieve(url_pt, "phishtank.csv")
    pt_df = pd.read_csv("phishtank.csv", usecols=['url'], nrows=2000)
    extra_phishing = pt_df['url'].dropna().tolist()
    phishing_urls.extend(extra_phishing[:2000])
    print(f"   ✓ Downloaded {len(extra_phishing)} PhishTank URLs!")
except:
    print("   PhishTank download failed — using built-in dataset")

print(f"   Total phishing URLs: {len(phishing_urls)}")
print(f"   Total safe URLs: {len(safe_urls)}")

# ─── FEATURE EXTRACTION ───────────────────────────────────────
print("\n[2/4] Extracting features...")

X, y = [], []
errors = 0

for url in phishing_urls:
    f = extract_features(url)
    if f:
        X.append(f)
        y.append(1)  # 1 = phishing
    else:
        errors += 1

for url in safe_urls:
    f = extract_features(url)
    if f:
        X.append(f)
        y.append(0)  # 0 = safe
    else:
        errors += 1

X = np.array(X)
y = np.array(y)

print(f"   Features extracted: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Phishing samples: {sum(y==1)}")
print(f"   Safe samples: {sum(y==0)}")
print(f"   Extraction errors: {errors}")

# ─── TRAIN MODEL ─────────────────────────────────────────────
print("\n[3/4] Training ML model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# Voting ensemble
model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

print("   Training ensemble model (RF + GradientBoosting)...")
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   ✓ Model trained successfully!")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# Feature importance
rf_model = model.estimators_[0]
importances = rf_model.feature_importances_
top_features = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:10]
print("\n   Top 10 Most Important Features:")
for feat, imp in top_features:
    bar = '█' * int(imp * 100)
    print(f"   {feat:<30} {bar} {imp*100:.1f}%")

# ─── SAVE MODEL ──────────────────────────────────────────────
print("\n[4/4] Saving model...")

joblib.dump(model, 'phishing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

model_info = {
    'accuracy': accuracy,
    'features': FEATURE_NAMES,
    'samples': len(y),
    'phishing': int(sum(y==1)),
    'safe': int(sum(y==0))
}
import json
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"   ✓ Saved: phishing_model.pkl")
print(f"   ✓ Saved: scaler.pkl")
print(f"   ✓ Saved: model_info.json")
print(f"\n{'='*60}")
print(f"   MODEL READY! Accuracy: {accuracy*100:.2f}%")
print(f"{'='*60}\n")