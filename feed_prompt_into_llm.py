from personalized_search import recommend_events, build_rag_prompt
import requests

# Replace with your actual Hugging Face API key
HUGGINGFACE_API_KEY = "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"

# Use a good, free model
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

def query_huggingface_llm(prompt, model=MODEL_NAME):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 300
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            return response.json()[0]['generated_text']
        except Exception as e:
            return "⚠️ Response parsing error: " + str(e)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return "Error contacting the model."

def extract_llm_answer(output: str) -> str:
    if "[INSIGHT]" in output:
        return output.split("[INSIGHT]")[-1].strip()
    return output.strip()

if __name__ == "__main__":
    user_id = 90
    query_text = "electronic pop concerts in İstanbul"

    # Step 1: Get events
    retrieved_events = recommend_events(user_id, query_text)

    # Step 2: Build RAG prompt
    prompt = build_rag_prompt(query_text, retrieved_events)

    print("\n📨 Sending RAG Prompt to LLM...\n")
    print(prompt)

    # Step 3: Query LLM
    llm_response = query_huggingface_llm(prompt)

    # Step 4: Debug: print raw model output
    print("\n📩 Raw LLM Full Response:\n")
    print(llm_response)

    # Step 5: Extract just the answer
    final_output = extract_llm_answer(llm_response)

    print("\n🧠 Extracted Insight:\n")
    print(final_output)
