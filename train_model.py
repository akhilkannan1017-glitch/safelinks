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

# ─── DATASET ──────────────────────────────────────────────────
print("\n[1/4] Loading dataset...")

# Built-in dataset for training
phishing_urls = [
    # Known phishing patterns
    "http://paypal-secure-login.tk/verify/account",
    "http://amazon-account-update.ml/confirm/password",
    "http://google-account-signin-verify.xyz/update",
    "http://apple-id-locked.cf/signin/confirm",
    "http://microsoft-support-alert.ga/helpdesk",
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
    "http://login-paypal.com.verify.xyz/webscr?cmd=login",
    "http://secure.amazon.com.verify-account.tk/ap/signin",
    "http://accounts.google.com.phishing.ml/ServiceLogin",
    "http://appleid.apple.com.locked.xyz/signin",
    "http://login.microsoftonline.com.phishing.tk/verify",
    "http://xn--pypal-4ve.com/verify",
    "http://paypa1.com/login",
    "http://arnazon.com/account",
    "http://micosoft.com/update",
    "http://gooogle.com/signin",
    "http://faceb00k.com/login",
    "http://secure-paypal-login.com/confirm/account/suspended/verify/password",
    "http://update-your-amazon-account.verify.credentials.xyz/signin",
    "http://your-apple-id-has-been-locked.confirm.ml/unlock",
    "http://account.google.com.verify.suspended.login.tk/security",
] + [
    "http://random" + str(i) + "phish.tk/login/verify" for i in range(50)
] + [
    "http://secure" + str(i) + "bank.ml/account/update/password" for i in range(50)
] + [
    "http://" + str(i) + "." + str(i) + "." + str(i) + ".1/login" for i in range(50)
]

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
    "https://www.django.rest-framework.org",
    "https://www.npmjs.com",
    "https://docs.google.com/document/d/1abc",
    "https://mail.google.com/mail/u/0",
    "https://drive.google.com/file/d/abc",
    "https://accounts.google.com/signin",
    "https://myaccount.google.com",
    "https://support.microsoft.com/en-us",
    "https://office.microsoft.com/en-us",
    "https://login.microsoftonline.com",
    "https://portal.azure.com",
    "https://docs.microsoft.com/en-us",
] + [
    "https://www.site" + str(i) + ".com/page/about" for i in range(100)
] + [
    "https://blog" + str(i) + ".wordpress.com/2024/article" for i in range(50)
] + [
    "https://shop" + str(i) + ".shopify.com/products" for i in range(50)
]

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