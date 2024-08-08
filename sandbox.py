from transformers import AutoTokenizer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load a pre-trained tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", add_bos_token=True)

# Define the sentence
sentence = "This is an example sentence."

# Tokenize the sentence
encoded = tokenizer.encode(sentence)
encoded_tokens = tokenizer.convert_ids_to_tokens(encoded)
encoded_ids = encoded

print(f"Encoded Tokens: {encoded_tokens}")
print(f"Encoded IDs: {encoded_ids}")

# Decode the tokens back to the sentence
decoded = tokenizer.decode(encoded_ids)
print(f"Decoded Sentence: {decoded}")




