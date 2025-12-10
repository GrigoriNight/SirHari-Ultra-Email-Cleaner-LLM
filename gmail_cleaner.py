# sirhari_ultra_llm.py
# Auto-installs dependencies + Full AI Email Brain + Personal LLM

import os
import sys
import time
import threading
import pickle
import logging
import random
import shutil
import urllib.request
import json
import base64
import subprocess
from datetime import datetime

# ---------------- AUTO-INSTALL DEPENDENCIES ----------------
def install_package(package):
    print(f"[INSTALL] Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

def ensure_dependencies():
    required = [
        "pyttsx3",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "google-auth",
        "google-auth-oauthlib",
        "google-api-python-client",
        "PyQt5"
    ]
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            install_package(pkg)

ensure_dependencies()

# Now import everything
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

import pyttsx3
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit,
    QLabel, QLineEdit, QProgressBar, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QTextCursor, qRegisterMetaType

qRegisterMetaType(QTextCursor)

# ---------------- CONFIG ----------------
SCOPES = ['https://www.googleapis.com/auth/gmail.modify', 'https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
MODEL_FILE = 'sirhari_model.pt'
LLM_DIR = './sirhari_llm'
DATASET_FILE = 'email_data.jsonl'
DO_NOT_DELETE_FILE = 'DO_NOT_DELETE.txt'
ENGLISH_WORDS_FILE = 'english_words.txt'
SAFE_LABEL_NAME = 'SirHari_Safe'
BATCH_FETCH = 50
USE_WORDLIST = True
AGGRESSIVE_DELETE = False
AUTO_DOWNLOAD_WORDLIST = True

SEED_KEYWORDS = [
    'free', 'winner', 'offer', 'click here', 'urgent', 'promotion',
    'lottery', 'royalty free', 'guaranteed', 'limited time', 'act now'
]
FEATURE_SIZE = 3 + len(SEED_KEYWORDS)

logging.basicConfig(
    filename='sirhari.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------- UTILS ----------------
def speak(text):
    def _say():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"TTS failed: {e}")
    threading.Thread(target=_say, daemon=True).start()

def load_do_not_delete():
    if not os.path.exists(DO_NOT_DELETE_FILE):
        with open(DO_NOT_DELETE_FILE, 'w', encoding='utf-8') as f:
            f.write("# One rule per line:\n# boss@company.com\n# @bank.com\n# invoice\n")
        return []
    with open(DO_NOT_DELETE_FILE, 'r', encoding='utf-8') as f:
        return [l.strip().lower() for l in f if l.strip() and not l.startswith('#')]

def download_wordlist():
    if os.path.exists(ENGLISH_WORDS_FILE):
        return
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    try:
        print(f"[DOWNLOAD] Fetching wordlist...")
        urllib.request.urlretrieve(url, ENGLISH_WORDS_FILE)
        print(f"[INFO] Downloaded {ENGLISH_WORDS_FILE}")
    except Exception as e:
        print(f"[WARN] Wordlist download failed: {e}")

def load_wordlist():
    if not USE_WORDLIST or not os.path.exists(ENGLISH_WORDS_FILE):
        return set()
    with open(ENGLISH_WORDS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        return {w.strip().lower() for w in f if w.strip() and len(w.strip()) > 2}

DO_NOT_DELETE = load_do_not_delete()
WORDSET = load_wordlist()

def is_protected(sender, subject):
    s = (sender or '').lower()
    subj = (subject or '').lower()
    for rule in DO_NOT_DELETE:
        r = rule.lower()
        if r.startswith('@') and s.endswith(r): return True
        if '@' in r and s == r: return True
        if r in subj: return True
    return False

def extract_basic_features(subject, sender):
    subj = (subject or '').lower()
    snd = (sender or '').lower()
    subj_len = float(len(subj))
    sender_digits = float(sum(c.isdigit() for c in snd))
    word_frac = 0.0
    if WORDSET:
        words = [w.strip('.,!?:;()[]') for w in subj.split() if w.strip('.,!?:;()[]')]
        if words:
            matches = sum(1 for w in words if w in WORDSET)
            word_frac = matches / len(words)
    seed_feat = [1.0 if k in subj or k in snd else 0.0 for k in SEED_KEYWORDS]
    feat = [subj_len, sender_digits, word_frac] + seed_feat
    assert len(feat) == FEATURE_SIZE
    return feat

# ---------------- MODEL ----------------
class SpamNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.out(self.fc3(x))

# ---------------- GMAIL AUTH ----------------
def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except:
                creds = None
        if not creds:
            if not os.path.exists(CREDENTIALS_FILE):
                QMessageBox.critical(None, "Missing", f"Download {CREDENTIALS_FILE} from Google Cloud Console.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)
    return build('gmail', 'v1', credentials=creds)

# ---------------- GMAIL HELPERS ----------------
def safe_execute(call, *args, **kwargs):
    backoff = 1
    while True:
        try:
            return call(*args, **kwargs).execute()
        except HttpError as e:
            if e.resp.status in (429, 500, 503):
                wait = backoff + random.uniform(0, 2)
                logging.warning(f"Rate limit. Wait {wait:.1f}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            else:
                raise
        except Exception as e:
            logging.error(f"API error: {e}")
            time.sleep(2)

def get_or_create_label(service, label_name=SAFE_LABEL_NAME):
    labels = safe_execute(service.users().labels().list, userId='me').get('labels', [])
    for lbl in labels:
        if lbl.get('name') == label_name:
            return lbl.get('id')
    body = {"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
    created = safe_execute(service.users().labels().create, userId='me', body=body)
    return created.get('id')

def fetch_message_metadata(service, msg_id):
    try:
        msg = safe_execute(service.users().messages().get, userId='me', id=msg_id, format='metadata', metadataHeaders=['From', 'Subject'])
        headers = msg.get('payload', {}).get('headers', [])
        sender = subject = ''
        for h in headers:
            name = h.get('name', '').lower()
            if name == 'from': sender = h.get('value', '')
            if name == 'subject': subject = h.get('value', '')
        return sender, subject
    except Exception as e:
        logging.error(f"Metadata {msg_id}: {e}")
        return '', ''

# ---------------- GUI SIGNALS ----------------
class Signals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, int, int)

# ---------------- MAIN APP ----------------
class SirHariApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SirHari Ultra LLM — Zero Setup AI")
        self.setGeometry(100, 100, 1000, 760)
        self.setup_ui()

        self.signals = Signals()
        self.signals.log.connect(self.append_log)
        self.signals.progress.connect(self.update_progress)

        self.stop_flag = threading.Event()
        self.model = SpamNet(FEATURE_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()
        self.service = None
        self.safe_label_id = None

        self.load_model()
        speak("Sir Hari online. All systems ready.")

        if AUTO_DOWNLOAD_WORDLIST and USE_WORDLIST and not os.path.exists(ENGLISH_WORDS_FILE):
            threading.Thread(target=download_wordlist, daemon=True).start()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.chat_input = QLineEdit()
        self.chat_btn = QPushButton("Ask SirHari (Your AI)")
        self.dry_btn = QPushButton("Dry Run (Train Spam)")
        self.del_btn = QPushButton("DELETE SPAM")
        self.stop_btn = QPushButton("STOP")
        self.undo_btn = QPushButton("Undo Last")
        self.train_llm_btn = QPushButton("Train AI on My Emails")

        layout.addWidget(QLabel("<b>SirHari Ultra Log:</b>"))
        layout.addWidget(self.log_box)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("<b>Chat with Your Personal AI:</b>"))
        layout.addWidget(self.chat_input)
        layout.addWidget(self.chat_btn)
        layout.addWidget(self.dry_btn)
        layout.addWidget(self.del_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.undo_btn)
        layout.addWidget(self.train_llm_btn)
        self.setLayout(layout)

        self.chat_btn.clicked.connect(self.chat)
        self.dry_btn.clicked.connect(self.start_dry_run)
        self.del_btn.clicked.connect(self.confirm_delete)
        self.stop_btn.clicked.connect(self.stop_process)
        self.undo_btn.clicked.connect(self.undo_last)
        self.train_llm_btn.clicked.connect(self.train_on_emails)

    def append_log(self, txt):
        self.log_box.append(txt)
        logging.info(txt)

    def update_progress(self, seen, skipped, deleted, total):
        if total:
            pct = int((seen + skipped) / total * 100)
            self.progress_bar.setValue(pct)
        self.append_log(f"[PROGRESS] Seen:{seen} Safe:{skipped} Deleted:{deleted} ~{total-seen} left")

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
                self.append_log("[INFO] Spam model loaded.")
            except Exception as e:
                self.append_log(f"[WARN] Load failed: {e}")
        else:
            self.append_log("[INFO] New spam model created.")

    def save_model(self):
        try:
            backup = f"{MODEL_FILE}.bak.{int(time.time())}"
            if os.path.exists(MODEL_FILE):
                shutil.copy(MODEL_FILE, backup)
            torch.save(self.model.state_dict(), MODEL_FILE)
            self.append_log(f"[INFO] Model saved.")
        except Exception as e:
            self.append_log(f"[ERROR] Save failed: {e}")

    def start_dry_run(self):
        self.stop_flag.clear()
        threading.Thread(target=self.run_cleaner, args=(True,), daemon=True).start()

    def confirm_delete(self):
        reply = QMessageBox.question(
            self, "CONFIRM DELETE",
            "Spam will go to Trash (recoverable 30 days).\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stop_flag.clear()
            threading.Thread(target=self.run_cleaner, args=(False,), daemon=True).start()

    def stop_process(self):
        self.stop_flag.set()
        self.append_log("[STOP] Stopping…")

    def undo_last(self):
        if not self.service:
            self.service = authenticate()
        try:
            res = safe_execute(self.service.users().messages().list, userId='me', labelIds=['TRASH'], maxResults=10)
            msgs = res.get('messages', [])
            if not msgs:
                speak("Trash is empty.")
                return
            latest = msgs[0]['id']
            safe_execute(self.service.users().messages().modify, userId='me', id=latest, body={'removeLabelIds': ['TRASH']})
            sender, subj = fetch_message_metadata(self.service, latest)
            self.append_log(f"[UNDO] Restored: {subj}")
            speak("Last delete undone.")
        except Exception as e:
            self.append_log(f"[ERROR] Undo: {e}")

    # ---------------- LLM: DATA ----------------
    def collect_email_data(self, max_emails=500):
        if not self.service:
            self.service = authenticate()
        data = []
        try:
            res = safe_execute(self.service.users().messages().list, userId='me', maxResults=max_emails)
            msgs = res.get('messages', [])
            self.append_log(f"[COLLECT] Processing {len(msgs)} emails...")
            for m in msgs:
                msg_id = m['id']
                full_msg = safe_execute(self.service.users().messages().get, userId='me', id=msg_id, format='full')
                payload = full_msg.get('payload', {})
                headers = payload.get('headers', [])
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')

                if is_protected(sender, subject):
                    continue

                body = ''
                parts = payload.get('parts', [payload])
                for part in parts:
                    if part.get('mimeType') == 'text/plain':
                        data_b64 = part.get('body', {}).get('data', '')
                        if data_b64:
                            body = base64.urlsafe_b64decode(data_b64).decode('utf-8', errors='ignore')
                            break

                if subject and body.strip():
                    data.append({
                        "prompt": f"Subject: {subject} From: {sender}",
                        "completion": body.strip()[:1000]
                    })
        except Exception as e:
            self.append_log(f"[ERROR] Collection: {e}")

        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        self.append_log(f"[COLLECT] Saved {len(data)} samples.")
        return DATASET_FILE

    # ---------------- LLM: TRAIN ----------------
    def train_on_emails(self):
        def _train():
            try:
                self.collect_email_data(max_emails=500)
                if not os.path.exists(DATASET_FILE):
                    return

                from datasets import load_dataset
                from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

                dataset = load_dataset('json', data_files=DATASET_FILE, split='train')
                model_name = "distilgpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token

                def tokenize(examples):
                    texts = [p + tokenizer.eos_token + c for p, c in zip(examples['prompt'], examples['completion'])]
                    tokenized = tokenizer(texts, truncation=True, max_length=256, padding='max_length')
                    tokenized['labels'] = tokenized['input_ids'].copy()
                    return tokenized

                tokenized = dataset.map(tokenize, batched=True)
                model = AutoModelForCausalLM.from_pretrained(model_name)

                args = TrainingArguments(
                    output_dir=LLM_DIR,
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    save_steps=200,
                    save_total_limit=2,
                    logging_dir='./logs',
                    fp16=torch.cuda.is_available(),
                    report_to=[],
                )

                trainer = Trainer(model=model, args=args, train_dataset=tokenized)
                self.append_log("[TRAINING] Learning your email style...")
                trainer.train()
                model.save_pretrained(LLM_DIR)
                tokenizer.save_pretrained(LLM_DIR)
                self.append_log("[DONE] Your AI is now trained on your emails!")
                speak("I learned from your emails. Ask me anything.")
            except Exception as e:
                self.append_log(f"[ERROR] Training: {e}")

        threading.Thread(target=_train, daemon=True).start()

    # ---------------- CHAT ----------------
    def chat(self):
        q = self.chat_input.text().strip()
        if not q: return
        self.append_log(f"[USER] {q}")
        self.chat_input.clear()

        def respond():
            try:
                if not os.path.exists(LLM_DIR):
                    reply = "Train me first on your emails!"
                else:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
                    model = AutoModelForCausalLM.from_pretrained(LLM_DIR)
                    inputs = tokenizer(q, return_tensors="pt")
                    outputs = model.generate(**inputs, max_length=120, temperature=0.7, do_sample=True)
                    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).split(q, 1)[-1].strip()
                    if not reply:
                        reply = "I'm thinking..."
                self.append_log(f"[SIRHARI] {reply}")
                speak(reply)
            except Exception as e:
                self.append_log(f"[ERROR] Chat: {e}")
                reply = "Train me first!"
                self.append_log(f"[SIRHARI] {reply}")
                speak(reply)

        threading.Thread(target=respond, daemon=True).start()

    # ---------------- CLEANER ----------------
    def train_batch(self, X, y):
        if not X: return
        self.model.train()
        for _ in range(2):
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

    def run_cleaner(self, dry_run=True):
        try:
            self.service = authenticate()
        except Exception as e:
            self.append_log(f"[ERROR] Auth: {e}")
            return

        self.safe_label_id = get_or_create_label(self.service)
        self.append_log("[INFO] Safe label ready.")

        total_est = safe_execute(self.service.users().messages().list, userId='me', maxResults=1).get('resultSizeEstimate', 0)
        self.progress_bar.setMaximum(total_est if total_est else 0)

        page_token = None
        seen = skipped = deleted = 0

        while not self.stop_flag.is_set():
            try:
                resp = safe_execute(self.service.users().messages().list, userId='me', maxResults=BATCH_FETCH, pageToken=page_token)
            except Exception as e:
                self.append_log(f"[ERROR] List: {e}")
                break

            msgs = resp.get('messages', [])
            if not msgs: break

            to_delete = []
            X_batch, y_batch = [], []

            for m in msgs:
                if self.stop_flag.is_set(): break
                seen += 1
                msg_id = m['id']
                sender, subject = fetch_message_metadata(self.service, msg_id)

                if is_protected(sender, subject):
                    skipped += 1
                    self.append_log(f"[SAFE] {subject}")
                    try:
                        safe_execute(self.service.users().messages().modify, userId='me', id=msg_id, body={'addLabelIds': [self.safe_label_id]})
                    except: pass
                    continue

                feats = extract_basic_features(subject, sender)
                x = torch.tensor([feats], dtype=torch.float32)
                with torch.no_grad():
                    score = self.model(x).item()

                spam = (AGGRESSIVE_DELETE or score >= 0.80)
                label = 1.0 if spam else 0.0

                if spam and not dry_run:
                    to_delete.append(msg_id)
                elif dry_run and spam:
                    self.append_log(f"[DRY] Would delete: {subject} ({score:.2f})")

                X_batch.append(feats)
                y_batch.append(label)

            for mid in to_delete:
                if self.stop_flag.is_set(): break
                try:
                    safe_execute(self.service.users().messages().trash, userId='me', id=mid)
                    deleted += 1
                    self.append_log(f"[DELETED] {mid}")
                except Exception as e:
                    self.append_log(f"[ERROR] Delete {mid}: {e}")

            if X_batch:
                X = torch.tensor(X_batch, dtype=torch.float32)
                y = torch.tensor([[v] for v in y_batch], dtype=torch.float32)
                self.train_batch(X, y)

            self.save_model()
            self.signals.progress.emit(seen, skipped, deleted, total_est)
            page_token = resp.get('nextPageToken')
            if not page_token: break
            time.sleep(0.6)

        self.append_log(f"[DONE] Seen:{seen} Safe:{skipped} Deleted:{deleted}")
        speak("Cleanup complete.")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if AUTO_DOWNLOAD_WORDLIST and USE_WORDLIST:
        download_wordlist()

    app = QApplication(sys.argv)
    win = SirHariApp()
    win.show()
    sys.exit(app.exec_())