from datasets import Dataset, DatasetDict

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def create_pairs(km_sentences, vi_sentences):
    pairs = []
    for idx, (km, vi) in enumerate(zip(km_sentences, vi_sentences)):
        entry = {
            "id": str(idx),
            "translation": {
                "km": km,
                "vi": vi
            }
        }
        pairs.append(entry)
    return pairs

PATH = "final-project/data"

# Read data from file
train_km = read_file(f'{PATH}/train.km')
train_vi = read_file(f'{PATH}/train.vi')
dev_km = read_file(f'{PATH}/dev.km')
dev_vi = read_file(f'{PATH}/dev.vi')
test_km = read_file(f'{PATH}/test.km')
test_vi = read_file(f'{PATH}/test.vi')

# Create dataset
train_data = create_pairs(train_km, train_vi)
dev_data = create_pairs(dev_km, dev_vi)
test_data = create_pairs(test_km, test_vi)

# Create dataset
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)

# Create dataset dict
dataset = DatasetDict({
    'train': train_dataset,
    'dev': dev_dataset,
    'test': test_dataset
})