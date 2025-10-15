from transformers import AutoTokenizer

# Load Qwen tokenizer (adjust model name to the one you have)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt")
print(tokens["input_ids"])
print(tokens["input_ids"].shape)