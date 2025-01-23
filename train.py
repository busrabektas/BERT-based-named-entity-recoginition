import numpy as np
import argparse
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, BertConfig
from datasets import load_dataset
import ast
import evaluate

seqeval = evaluate.load("seqeval")

label_to_id = {
    "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6,
    "B-GEO": 7, "I-GEO": 8, "B-GPE": 9, "I-GPE": 10, "B-TIM": 11, "I-TIM": 12,
    "B-ART": 13, "I-ART": 14, "B-CUR": 15, "I-CUR": 16, "B-MISC": 17, "I-MISC": 18,
    "B-per": 1, "I-per": 2, "B-org": 3, "I-org": 4, "B-loc": 5, "I-loc": 6,
    "B-geo": 7, "I-geo": 8, "B-gpe": 9, "I-gpe": 10, "B-tim": 11, "I-tim": 12,
    "B-art": 13, "I-art": 14, "B-cur": 15, "I-cur": 16, "B-misc": 17, "I-misc": 18
}

label_list = []
for label in label_to_id.keys():
    label_list.append(label.upper())


def preprocess_tags(dataset):

    def convert_tags(tags):
        try:
            parsed_tags = ast.literal_eval(tags)
            flat_tags = []
            for tag in parsed_tags:
                if isinstance(tag, list):
                    flat_tags.extend(tag)
                else:
                    flat_tags.append(tag)
            return [label_to_id.get(tag, 0) for tag in flat_tags]
        except Exception as e:
            print(f"Error while parsing tags: {e}")
            raise

    dataset = dataset.map(lambda x: {"Tag": convert_tags(x["Tag"])})
    return dataset

def tokenize_and_align_labels(examples):

    sentences = [sentence.split() for sentence in examples['Sentence']]
    tokenized_inputs = tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=128,
        is_split_into_words=True
    )

    all_labels = []
    for i, label in enumerate(examples['Tag']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  
            elif word_idx < len(label):  
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],

    }


def main(args):
    raw_dataset = load_dataset('csv', data_files=args.dataset_path)['train']

    raw_dataset = preprocess_tags(raw_dataset)

    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dslim/bert-base-NER')
    config = BertConfig.from_pretrained('dslim/bert-base-NER', num_labels=len(label_to_id))
    model = BertForTokenClassification.from_pretrained(
        'dslim/bert-base-NER',
        config=config,
        ignore_mismatched_sizes=True
    )

    tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)

    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50, 
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=int(args.num_train_epoch),
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    print(f"Model and tokenizer saved at {args.model_save_path}")
    print(trainer.evaluate(eval_dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--num_train_epoch", type=int, required=True, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)