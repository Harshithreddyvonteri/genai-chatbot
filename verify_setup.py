import os
import sys

print("🔍 Verifying your environment...\n")

# 1. Check Python Version
print("1️⃣ Checking Python version...")
print(f"   Python executable: {sys.executable}")
print(f"   Python version   : {sys.version}\n")

# 2. Check Required Libraries
print("2️⃣ Checking required libraries...")
required_libs = [
    "streamlit",
    "chromadb",
    "google.generativeai",
    "sentence_transformers",
]

for lib in required_libs:
    try:
        __import__(lib if lib != "google.generativeai" else "google.generativeai as genai")
        print(f"   ✅ {lib} is installed.")
    except ImportError:
        print(f"   ❌ {lib} NOT installed. Run: pip install {lib}")

print()

# 3. Check GEMINI_API_KEY
print("3️⃣ Checking Gemini API key...")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print("   ✅ GEMINI_API_KEY is set.")
else:
    print("   ❌ GEMINI_API_KEY not found! Set it using:")
    print('      setx GEMINI_API_KEY "your_api_key_here"')
print()

# 4. Test Gemini API
print("4️⃣ Testing Gemini API...")
try:
    import google.generativeai as genai
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Hello! Can you confirm you are working?")
        print(f"   ✅ Gemini API response: {response.text}")
    else:
        print("   ⚠️ Skipped Gemini test (no API key).")
except Exception as e:
    print(f"   ❌ Gemini API test failed: {e}")
print()

# 5. Test ChromaDB
print("5️⃣ Testing ChromaDB...")
try:
    import chromadb
    db_path = "data/vectorstore/v2_chroma"
    if os.path.exists(db_path):
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        if collections:
            print(f"   ✅ ChromaDB loaded. Collections: {[c.name for c in collections]}")
        else:
            print("   ⚠️ No collections found in ChromaDB.")
    else:
        print(f"   ⚠️ ChromaDB path not found: {db_path}")
except Exception as e:
    print(f"   ❌ ChromaDB test failed: {e}")
print()

print("🎉 Verification complete!")
