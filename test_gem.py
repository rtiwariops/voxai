# test_gemini.py
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
MODEL   = os.getenv("GENAI_MODEL")

if not API_KEY:
    print("❌ GENAI_API_KEY is missing!", file=sys.stderr)
    sys.exit(1)

if not MODEL:
    print("❌ GENAI_MODEL is missing!", file=sys.stderr)
    sys.exit(1)

genai.configure(api_key=API_KEY)

try:
    chat = genai.GenerativeModel(MODEL).start_chat()
except Exception as e:
    print(f"❌ Failed to start chat with model '{MODEL}': {e}", file=sys.stderr)
    sys.exit(1)

prompt = "Hello, world! How are you today?"
try:
    # Remove the timeout=20; just do a plain send_message
    response = chat.send_message(prompt)
    print("✅ Gemini response (non‐streaming):")
    print(response.text)
except Exception as e:
    print(f"❌ chat.send_message failed: {e}", file=sys.stderr)
    sys.exit(1)
