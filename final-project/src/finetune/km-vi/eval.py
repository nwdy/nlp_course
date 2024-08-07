import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import dataset
from tqdm import tqdm
import sacrebleu


# TODO: Change this function to load your own data
# def load_file(file_path):
#     lang = []

#     with open(file_path, "r", encoding="utf-8") as file:
#         content_en = file.read()
#     lang += content_en.split('\n')
#     lang = [html.unescape(sent) for sent in lang]
#     return lang


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
username = "nwdy"
model_name = "fine-tuned-mbart-50-km-vi"
model_name = f"{username}/{model_name}"

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='km_KH', tgt_lang='vi_VN')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)
model.eval()


references = []
hypotheses = []

def eval(dataset):
    for example in tqdm(dataset['translation'], desc="Computing BLEU score"):
        inputs = tokenizer(example['km'], return_tensors="pt", max_length=64, truncation=True, 
                        padding="max_length").to(device)
        forced_bos_token_id = tokenizer.lang_code_to_id["vi_VN"]
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id,
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        hypotheses.append(prediction)
        references.append(example['vi'])
    
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    ter = sacrebleu.corpus_ter(hypotheses, references)
    chrf = sacrebleu.corpus_chrf(hypotheses, references)
    return bleu, ter, chrf


if __name__ == "__main__":
    bleu, ter, chrf = eval(dataset['test'][:20])
    print(f"BLEU: {bleu.score:.2f}")
    print(f"TER: {ter.score:.2f}")
    print(f"CHRF: {chrf.score:.2f}")


    # translation = {
    #     'inputs':[],
    #     'preds':[],
    #     'labels':[]
    # }

    # for i in range(len(list_test[SRC])):
    #     translation['inputs'].append(list_test[SRC][i])
    #     translation['preds'].append(predict(model, list_test[SRC][i], tokenizer))
    #     translation['labels'].append(list_test[TRG][i])

    # # Tính BLEU
    # bleu = sacrebleu.corpus_bleu(translation['preds'], [translation['labels']])
    # # Tính TER
    # ter = sacrebleu.corpus_ter(translation['preds'], [translation['labels']])
    # # Tính CHRF
    # chrf = sacrebleu.corpus_chrf(translation['preds'], [translation['labels']])

    # # Tính ROUGE
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # rouge1_scores = []
    # rougeL_scores = []
    # for pred, label in zip(translation['preds'], translation['labels']):
    #     scores = scorer.score(pred, label)
    #     rouge1_scores.append(scores['rouge1'].fmeasure)
    #     rougeL_scores.append(scores['rougeL'].fmeasure)

    # avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    # avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    



    # metric_str = f"bleu\tter\tchrf\trouge1\trougeL\n{bleu.score}\t{ter.score}\t{chrf.score}\t{avg_rouge1}\t{avg_rougeL}"

    # f = open('final-result\metric.txt', 'w', encoding='utf-8')
    # f.write(metric_str)
    # f.close()

    # pd.DataFrame(translation).to_csv('final-result/translation.csv', index=False)

    # print("Lưu thành công")