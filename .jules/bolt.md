## 2024-06-25 - Avoid repository pollution with fake-gcs-server
**Learning:** Extracting local testing binaries like `fake-gcs-server` in the repo root without care can accidentally overwrite critical project files (like `README.md` and `LICENSE`) and pollute git history if not properly `.gitignore`d or executed in `/tmp`.
**Action:** Always download/extract third-party tools into isolated `/tmp` directories, or be extremely careful to only extract specific binaries and explicitly `.gitignore` them and their logs.
