from transformers import MBart50TokenizerFast
from model.transformer import Transformer
from huggingface_hub import PyTorchModelHubMixin
import torch

class MyModel(Transformer, PyTorchModelHubMixin):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, 
                 d_ff, max_seq_length, dropout):
        super(MyModel, self).__init__(src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                                      num_layers, d_ff, max_seq_length, dropout)


class Translator:
    def __init__(self, src_lang, tgt_lang, model_name):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = MBart50TokenizerFast \
            .from_pretrained("facebook/mbart-large-50", src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = MyModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def translate(self, sentence):
        self.model.eval()
        self.model.to(self.device)

        # Tokenize the input sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding='max_length', 
                        max_length=64, truncation=True).to(self.device)
        src = inputs['input_ids']
        
        # Initialize the target sentence with the start token
        tgt = torch.tensor([self.tokenizer.lang_code_to_id["vi_VN"]] * 64) \
                    .unsqueeze(0).to(self.device)

        translated_sentence = []

        # Greedily decode the sentence
        with torch.no_grad():
            for i in range(64):
                output = self.model(src, tgt)
                next_token = output.argmax(dim=-1)[:, i]
                translated_sentence.append(next_token.item())

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                tgt[0, i+1] = next_token.item()

        # translated_sentence = translated_sentence[1:]  # Remove the start token
        translated_sentence = self.tokenizer.decode(translated_sentence, skip_special_tokens=True)
        
        return translated_sentence
    
if __name__ == "__main__":
    models = {
        "km-vi-base": {
            "model_name": "nwdy/transformer-km-vi",
            "src_lang": "km_KH",
            "tgt_lang": "vi_VN"
        },
        "vi-km-base": {
            "model_name": "nwdy/transformer-vi-km",
            "src_lang": "vi_VN",
            "tgt_lang": "km_KH"
        }
    }

    # Change the first index of the model dictionary to test different models
    model_name = models["km-vi-base"]["model_name"]
    src_lang = models["km-vi-base"]["src_lang"]
    tgt_lang = models["km-vi-base"]["tgt_lang"]

    translator = Translator(src_lang, tgt_lang, model_name)

    sentence = "មេរោគនេះងាយឆ្លង តែមិនឆ្លងដល់មនុស្សឡើយ។"
    # Expected: Bệnh cúm này rất dễ lây nhiễm nhưng không thể lây nhiễm sang người.
    pred = translator.translate(sentence)
    print(pred)
