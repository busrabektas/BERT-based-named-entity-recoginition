import json
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import pipeline
import argparse

def main(args):

    model = BertForTokenClassification.from_pretrained(args.model_load_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_load_path)

    label_to_id = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6,
        "B-GEO": 7, "I-GEO": 8, "B-GPE": 9, "I-GPE": 10, "B-TIM": 11, "I-TIM": 12,
        "B-ART": 13, "I-ART": 14, "B-CUR": 15, "I-CUR": 16, "B-MISC": 17, "I-MISC": 18,
        "B-per": 1, "I-per": 2, "B-org": 3, "I-org": 4, "B-loc": 5, "I-loc": 6,
        "B-geo": 7, "I-geo": 8, "B-gpe": 9, "I-gpe": 10, "B-tim": 11, "I-tim": 12,
        "B-art": 13, "I-art": 14, "B-cur": 15, "I-cur": 16, "B-misc": 17, "I-misc": 18
    }

    id_to_label = {v: k for k, v in label_to_id.items()}

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    with open("./data/test_input.txt", 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file if line.strip()]

    results = []

    for sentence in sentences:
        ner_results = ner_pipeline(sentence)
        entities = []
        
        for entity in ner_results:
            label = entity['entity_group']
            
            if label.startswith("LABEL_"):
                try:
                    label_id = int(label.split("_")[1])
                    label_text = id_to_label.get(label_id, label)  
                except (IndexError, ValueError):
                    label_text = label  
            else:
                label_text = label  
            
            entity_data = {
                "word": entity['word'],
                "entity_label": label_text,  
                "score": float(round(entity['score'], 4)),  
                "start": entity['start'],
                "end": entity['end']
            }
            entities.append(entity_data)
        
        sentence_result = {
            "sentence": sentence,
            "entities": entities
        }
        
        results.append(sentence_result)


    try:
        with open(args.output_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        print(f"NER results successfully written to '{args.output_file}' file")
    except TypeError as e:
        print(f"JSON serileştirme hatası: {e}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER pipeline")
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the trained model and tokenizer")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file containing sentences")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the NER results in JSON format")

    args = parser.parse_args()
    main(args)