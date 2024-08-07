import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "nwdy/fine-tuned-mbart-50-km-vi"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, 
                                                 src_lang="km_KH", tgt_lang="vi_VN")
model = MBartForConditionalGeneration.from_pretrained(model_name)
model.to(device)

def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, 
                       padding="max_length").to(device)
    forced_bos_token_id = tokenizer.lang_code_to_id["vi_VN"]
    outputs = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sentence = "មេរោគនេះងាយឆ្លង តែមិនឆ្លងដល់មនុស្សឡើយ។"
    # Expected: Bệnh cúm này rất dễ lây nhiễm nhưng không thể lây nhiễm sang người.
    pred = predict(sentence)
    print(pred)
