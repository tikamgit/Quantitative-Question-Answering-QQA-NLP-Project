import json

def extract_features(data):
    """Extract numerical values and keywords from questions."""
    relevant_keywords = ['friction', 'gravity', 'distance', 'speed', 'time', 'mass', 'force', 'energy', 'power', 'looks', 'appear', 'hear', 'strong', 'thick', 'rip']
    
    for item in data:
        numeric_values = item.get('numeric_values', [])
        question_lower = item['question'].lower()
        
        item['features'] = {
            'numeric_values': numeric_values,
            'num_numeric_values': len(numeric_values),
            'keywords': [word for word in question_lower.split() if word in relevant_keywords]
        }
    return data

if __name__ == "__main__":
    dataset_names = ['dev', 'test', 'train']
    for name in dataset_names:
        input_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}_processed.json'  # Adjust path as needed
        output_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}_features.json'
        print(f"Extracting features for {name} dataset...")
        with open(input_path, 'r') as f:
            data = json.load(f)
        processed_data = extract_features(data)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"Saved {name} dataset with features to {output_path}")