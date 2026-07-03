"""Diagnostic script — checks .env values for common formatting problems."""
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("WATSONX_APIKEY", "")
url = os.getenv("WATSONX_URL", "")
pid = os.getenv("WATSONX_PROJECT_ID", "")

print("=== .env diagnostic ===")
print(f"WATSONX_URL        : {url}")
print(f"WATSONX_PROJECT_ID : {pid}")
print(f"API key length     : {len(key)} chars")
print(f"API key prefix     : {key[:8]}..." if len(key) > 8 else f"API key value      : '{key}' (too short!)")
print(f"Has leading space  : {key != key.lstrip()}")
print(f"Has trailing space : {key != key.rstrip()}")
print(f'Has double-quotes  : {key.startswith(chr(34)) or key.endswith(chr(34))}')
print(f"Has single-quotes  : {key.startswith(chr(39)) or key.endswith(chr(39))}")
print()

issues = []
if not key or key == "your_ibm_cloud_api_key_here":
    issues.append("WATSONX_APIKEY is still the placeholder value")
if key != key.strip():
    issues.append("WATSONX_APIKEY has leading/trailing whitespace — remove it")
if key.startswith(('"', "'")):
    issues.append("WATSONX_APIKEY is wrapped in quotes — remove them")
if len(key) < 30:
    issues.append(f"WATSONX_APIKEY looks too short ({len(key)} chars) — IBM keys are typically 44 chars")
if not url.startswith("https://"):
    issues.append("WATSONX_URL does not start with https://")
if not pid or pid == "your_watsonx_project_id_here":
    issues.append("WATSONX_PROJECT_ID is still the placeholder value")

if issues:
    print("PROBLEMS FOUND:")
    for i in issues:
        print(f"  PROBLEM: {i}")
else:
    print("No obvious formatting issues found.")
    print("The key looks structurally correct — trying IAM token exchange now...")
    try:
        from ibm_watsonx_ai import Credentials, APIClient
        creds = Credentials(api_key=key, url=url)
        client = APIClient(credentials=creds, project_id=pid)
        print("SUCCESS — credentials are valid and IAM exchange worked.")
    except Exception as e:
        print(f"FAILED — {e}")
