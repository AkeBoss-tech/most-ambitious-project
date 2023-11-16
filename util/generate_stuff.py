print("Starting...")

from summarizer import Summarizer,TransformerSummarizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, XLNetTokenizer, XLNetLMHeadModel
import torch

print("Done importing...")

body = '''Calculus Parody of Havana
Original:
Hey
Havana, ooh na-na (ayy)

Calculus Version:
'''

""" GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=60))
print(full)

model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full) """

print("Obtaining Models...")
# GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# XLNet
xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")

def get_next_word_gpt2(input_text, additional_tokens=6, num_return_sequences=3):
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = [gpt2_model.generate(input_ids, max_length=len(input_ids[0]) + additional_tokens, num_return_sequences=1, temperature=5) for _ in range(num_return_sequences)]
    decoded_outputs = [[gpt2_tokenizer.decode(seq[0], skip_special_tokens=True) for seq in output] for output in outputs]
    next_words = [seqs[0][len(input_text):] for seqs in decoded_outputs]
    return next_words, decoded_outputs

def get_next_word_xlnet(input_text, additional_tokens=20):
    input_ids = xlnet_tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = xlnet_model.generate(input_ids, max_length=len(input_ids[0]) + additional_tokens, num_return_sequences=1, temperature=5)
    decoded_output = xlnet_tokenizer.decode(output[0], skip_special_tokens=True)
    next_word = decoded_output[len(input_text):]
    return next_word, decoded_output

print("Running Models")
def generate_text_with_gpt2(input_text, additional_tokens=10, num_return_sequences=10, temperature=0.4):
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = gpt2_tokenizer.pad_token_id

    with torch.no_grad():
        outputs = gpt2_model.generate(input_ids, attention_mask=attention_mask, pad_token_id=gpt2_tokenizer.eos_token_id, max_length=len(input_ids[0]) + additional_tokens, num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True)
    decoded_outputs = [gpt2_tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
    return [x[len(input_text):] for x in decoded_outputs]

generated_texts = generate_text_with_gpt2(body)

for text in generated_texts:
    print(text)

# print(*get_next_word_xlnet(body))
