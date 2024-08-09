import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

class Translator:
    def __init__(self, src_lang, tgt_lang, model_name):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = MBart50TokenizerFast \
            .from_pretrained("facebook/mbart-large-50", src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def translate(self, text) -> str:
        self.model.to(self.device)
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True, 
                                padding="max_length").to(self.device)
        forced_bos_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
        outputs = self.model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id,
        )
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred


if __name__ == "__main__":
    # Two fine-tuned models for Vietnamese-Khmer translation
    models = {
        "km-vi-ft": {
            "model_name": "nwdy/fine-tuned-mbart-50-km-vi",
            "src_lang": "km_KH",
            "tgt_lang": "vi_VN"
        },
        "vi-km-ft": {
            "model_name": "nwdy/fine-tuned-mbart-50-vi-km",
            "src_lang": "vi_VN",
            "tgt_lang": "km_KH"
        },
    }

    # Change the first index of the model dictionary to test different models
    model_name = models["km-vi-ft"]["model_name"]
    src_lang = models["km-vi-ft"]["src_lang"]
    tgt_lang = models["km-vi-ft"]["tgt_lang"]

    translator = Translator(src_lang, tgt_lang, model_name)

    sentence = "មេរោគនេះងាយឆ្លង តែមិនឆ្លងដល់មនុស្សឡើយ។"
    # Expected: Bệnh cúm này rất dễ lây nhiễm nhưng không thể lây nhiễm sang người.
    pred = translator.translate(sentence)
    print(pred)
