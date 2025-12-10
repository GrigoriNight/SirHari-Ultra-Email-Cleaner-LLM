import os
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox
from typing import List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If you modify these scopes, delete the file token.json.
SCOPES = ['https://mail.google.com/']


class GmailDeleterApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Gmail Deleter (safer)')
        self.root.geometry('720x480')

        # Controls frame
        ctrl_frame = tk.Frame(root)
        ctrl_frame.pack(pady=8)

        self.login_btn = tk.Button(ctrl_frame, text='Login to Gmail', command=self.login)
        self.login_btn.grid(row=0, column=0, padx=6)

        self.delete_btn = tk.Button(ctrl_frame, text='Delete All Unread Emails', command=self.delete_emails, state=tk.DISABLED)
        self.delete_btn.grid(row=0, column=1, padx=6)

        self.dry_run_var = tk.IntVar(value=1)
        self.dry_run_cb = tk.Checkbutton(ctrl_frame, text='Dry-run (do not actually delete)', variable=self.dry_run_var)
        self.dry_run_cb.grid(row=0, column=2, padx=6)

        self.limit_label = tk.Label(ctrl_frame, text='Limit (0 = all):')
        self.limit_label.grid(row=0, column=3)
        self.limit_entry = tk.Entry(ctrl_frame, width=6)
        self.limit_entry.insert(0, '0')
        self.limit_entry.grid(row=0, column=4, padx=6)

        # Log box
        self.log_box = scrolledtext.ScrolledText(root, width=92, height=24)
        self.log_box.pack(pady=8)

        self.service = None
        self.creds = None
        self._lock = threading.Lock()

    def log(self, text: str):
        # Schedule UI-safe logging
        def _append():
            self.log_box.insert(tk.END, text + "\n")
            self.log_box.see(tk.END)

        self.root.after(0, _append)

    def _set_buttons(self, enabled: bool):
        def _set():
            state = tk.NORMAL if enabled else tk.DISABLED
            self.login_btn.config(state=state)
            # keep delete button enabled only if logged in
            self.delete_btn.config(state=state if self.service else tk.DISABLED)

        self.root.after(0, _set)

    def login(self):
        self.log('Starting Gmail login...')
        self._set_buttons(False)
        threading.Thread(target=self._login_thread, daemon=True).start()

    def _login_thread(self):
        try:
            creds = None
            if os.path.exists('token.json'):
                try:
                    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                except Exception:
                    creds = None

            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    self.log('Refreshing access token...')
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        self.log(f'Failed to refresh token: {e}')
                        creds = None

                if not creds:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save the credentials for the next run
                with open('token.json', 'w') as token_file:
                    token_file.write(creds.to_json())

            self.creds = creds
            self.service = build('gmail', 'v1', credentials=creds)
            self.log('Login successful!')
            self._set_buttons(True)
        except FileNotFoundError:
            self.log('Missing credentials.json. Place the OAuth client secrets in credentials.json')
            messagebox.showerror('Error', 'Missing credentials.json. See README.')
            self._set_buttons(True)
        except Exception as e:
            self.log(f'Login failed: {e}')
            messagebox.showerror('Error', f'Login failed: {e}')
            self._set_buttons(True)

    def delete_emails(self):
        if not self.service:
            self.log('Not logged in!')
            return
        try:
            limit = int(self.limit_entry.get())
            if limit < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror('Error', 'Limit must be a non-negative integer')
            return

        self._set_buttons(False)
        threading.Thread(target=self._delete_thread, args=(limit,), daemon=True).start()

    def _collect_unread_ids(self, limit: int) -> List[str]:
        self.log('Searching for unread emails...')
        ids = []
        page_token = None
        try:
            while True:
                resp = self.service.users().messages().list(userId='me', q='is:unread', pageToken=page_token, maxResults=500).execute()
                msgs = resp.get('messages', [])
                for m in msgs:
                    ids.append(m['id'])
                    if limit and len(ids) >= limit:
                        return ids

                page_token = resp.get('nextPageToken')
                if not page_token:
                    break
        except HttpError as e:
            self.log(f'API error while listing messages: {e}')
        except Exception as e:
            self.log(f'Unexpected error while listing messages: {e}')

        return ids

    def _batch_delete(self, ids: List[str]):
        # Gmail batchDelete accepts a body of {'ids': [id1,id2,...]}
        chunk_size = 100
        total_deleted = 0
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            attempt = 0
            while attempt < 5:
                try:
                    self.service.users().messages().batchDelete(userId='me', body={'ids': chunk}).execute()
                    total_deleted += len(chunk)
                    self.log(f'Deleted batch {i // chunk_size + 1}: {len(chunk)} messages')
                    break
                except HttpError as e:
                    code = getattr(e, 'status_code', None) or getattr(e, 'resp', {}).get('status') if hasattr(e, 'resp') else None
                    attempt += 1
                    wait = 2 ** attempt
                    self.log(f'Batch delete failed (attempt {attempt}) with error: {e}. Retrying in {wait}s')
                    time.sleep(wait)
                except Exception as e:
                    self.log(f'Unexpected error during batch delete: {e}')
                    return total_deleted

        return total_deleted

    def _delete_thread(self, limit: int):
        try:
            ids = self._collect_unread_ids(limit)
            total_found = len(ids)
            self.log(f'Found {total_found} unread emails')

            if total_found == 0:
                self.log('No unread emails to delete.')
                self._set_buttons(True)
                return

            # Confirm with user
            if self.dry_run_var.get():
                msg = f'DRY RUN: Would delete {total_found} unread emails. Proceed?'
            else:
                msg = f'About to delete {total_found} unread emails. This is permanent. Proceed?'

            proceed = messagebox.askyesno('Confirm delete', msg)
            if not proceed:
                self.log('Deletion cancelled by user.')
                self._set_buttons(True)
                return

            if self.dry_run_var.get():
                # In dry-run, just list a sample of IDs
                sample = ids[:20]
                self.log('Dry-run mode. Sample message IDs:')
                for mid in sample:
                    self.log(f'  {mid}')
                self.log('No messages were deleted (dry-run).')
                self._set_buttons(True)
                return

            # Perform batch deletes
            deleted = self._batch_delete(ids)
            self.log(f'Deletion process completed. Total deleted: {deleted}')
        except Exception as e:
            self.log(f'An unexpected error occurred: {e}')
        finally:
            self._set_buttons(True)


def main():
    root = tk.Tk()
    app = GmailDeleterApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()