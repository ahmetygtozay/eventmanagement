from huggingface_hub import InferenceClient

client = InferenceClient(token="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")

prompt = "Hello world"

response = client(model=model, inputs=prompt, parameters={"max_new_tokens": 200, "temperature": 0.7})


print(response.generated_text)
