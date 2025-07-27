import json
import re

def load_data(file_path):
    """Load a JSON dataset from the specified file path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_scientific_numbers(text):
    """Extract numbers in scientific notation from text and convert them to floats."""
    pattern = r'\b\d+(?:\.\d*)?E[+-]?\d+\b'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]

def preprocess_data(data):
    """Preprocess the dataset by adding a 'numeric_values' field to each entry."""
    for item in data:
        if 'question_sci_10E' in item:
            numbers = extract_scientific_numbers(item['question_sci_10E'])
            item['numeric_values'] = numbers if numbers else []
            if not numbers:
                print(f"Warning: No numbers extracted from 'question_sci_10E' in item: {item.get('question', 'Unknown')}")
        else:
            print(f"Warning: 'question_sci_10E' missing in item: {item.get('question', 'Unknown')}")
            item['numeric_values'] = []
    return data

if __name__ == "__main__":
    dataset_names = ['dev', 'test', 'train']
    for name in dataset_names:
        input_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}.json'
        output_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}_processed.json'
        print(f"Processing {name} dataset...")
        data = load_data(input_path)
        processed_data = preprocess_data(data)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"Saved processed {name} dataset to {output_path}")