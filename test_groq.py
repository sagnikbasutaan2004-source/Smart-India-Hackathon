import os, requests, json, time

# Load .env
with open('.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

groq_key = os.environ.get('GROQ_API_KEY', '')
print(f"Key present: {bool(groq_key)} (prefix: {groq_key[:10]}...)\n")

body = {
    "model": "openai/gpt-oss-120b",
    "response_format": {"type": "json_object"},
    "temperature": 0.2,
    "max_tokens": 200,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always respond with pure JSON."
        },
        {
            "role": "user",
            "content": "Return a JSON object with key 'status' set to 'ok' and key 'message' set to 'Groq LLM is working correctly'."
        }
    ]
}

t0 = time.time()
resp = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    json=body,
    headers={"Authorization": f"Bearer {groq_key}"},
    timeout=60
)
elapsed = time.time() - t0

print(f"Status:  {resp.status_code}")
print(f"Latency: {elapsed:.2f}s")

if resp.status_code == 200:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"Model:   {data['model']}")
    print(f"Response JSON: {content}")
    print(f"Tokens:  prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
    parsed = json.loads(content)
    print(f"Parsed:  {parsed}")
    print()
    print("SUCCESS - Groq LLM is working correctly!")
else:
    print(f"FAILED - ERROR: {resp.text}")
