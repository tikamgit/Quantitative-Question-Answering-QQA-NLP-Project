import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import Dataset
import torch

def extract_numbers_from_option(option):
    """Extract numeric values from the option text."""
    numbers = re.findall(r'\d+\.?\d*', option)
    return [float(num) for num in numbers]

def predict_answer(item):
    """Rule-based prediction based on numeric values and question context."""
    features = item.get('features', {})
    numeric_values = features.get('numeric_values', [])
    question = item['question'].lower()
    options = [item['Option1'], item['Option2']]
    
    opt1_values = extract_numbers_from_option(options[0])
    opt2_values = extract_numbers_from_option(options[1])
    
    if 'speed' in question and 'time' in question and 'distance' in question:
        if len(numeric_values) >= 2:
            speed = numeric_values[0]
            time = numeric_values[1]
            calculated_distance = speed * time
            if opt1_values and opt2_values:
                diff1 = abs(calculated_distance - opt1_values[0])
                diff2 = abs(calculated_distance - opt2_values[0])
                return "Option 1" if diff1 < diff2 else "Option 2"

    elif 'friction' in question or 'heat' in question or 'hotter' in question:
        if len(numeric_values) >= 2:
            return "Option 2" if numeric_values[1] > numeric_values[0] else "Option 1"
        
    elif 'looks' in question or 'appear' in question or 'hear' in question:
        if len(numeric_values) >= 2:
            return "Option 1" if numeric_values[0] < numeric_values[1] else "Option 2"

    elif 'strong' in question or 'thick' in question or 'rip' in question:
        if len(numeric_values) >= 2:
            return "Option 1" if numeric_values[0] > numeric_values[1] else "Option 2"

    if numeric_values and opt1_values and opt2_values:
        diff1 = abs(numeric_values[0] - opt1_values[0])
        diff2 = abs(numeric_values[0] - opt2_values[0])
        return "Option 1" if diff1 < diff2 else "Option 2"

    return "Option 1"

def run_rule_based(data):
    """Run the rule-based baseline."""
    predictions = []
    for item in data:
        pred = predict_answer(item)
        predictions.append({
            'question': item['question'],
            'prediction': pred,
            'answer': item['answer']
        })
    return predictions

def convert_to_multiple_choice_format(data):
    """Convert data to a format suitable for multiple-choice training."""
    questions = []
    choices = []
    labels = []
    for item in data:
        questions.append(item['question'])
        choices.append([item['Option1'], item['Option2']])
        labels.append(0 if item['answer'] == "Option 1" else 1)
    return Dataset.from_dict({
        'questions': questions,
        'choices': choices,
        'labels': labels
    })

def fine_tune_model(train_data):
    """Fine-tune the pre-trained model on the training data."""
    train_dataset = convert_to_multiple_choice_format(train_data)
    
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    def preprocess_function(examples):
        first_sentences = [[q] * 2 for q in examples['questions']]
        second_sentences = [[c[0], c[1]] for c in examples['choices']]
        
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding=True)
        return {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

import os
from transformers import AutoTokenizer, AutoModelForMultipleChoice
def run_model(data, use_fine_tuned=True):
    if use_fine_tuned:
        model_path = os.path.abspath("./fine_tuned_model")
    else:
        model_path = "roberta-base"  # Pre-trained model from Hugging Face
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMultipleChoice.from_pretrained(model_path)
    
    predictions = []
    for item in data:
        question = item['question']
        options = [item['Option1'], item['Option2']]
        
        inputs = tokenizer(
            [question, question],
            options,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids'].unsqueeze(0)
        attention_mask = inputs['attention_mask'].unsqueeze(0)
        
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        with torch.no_grad():
            outputs = model(**model_inputs)
        
        logits = outputs.logits
        pred_index = logits.argmax(dim=1).item()
        
        predictions.append({
            'question': question,
            'prediction': f"Option {pred_index + 1}",
            'answer': item['answer']
        })
    
    return predictions

def run_zero_shot(data):
    """Run the zero-shot learning baseline."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    predictions = []
    for item in data:
        options = [item['Option1'], item['Option2']]
        result = classifier(item['question'], options)
        pred_index = options.index(result['labels'][0])
        pred = f"Option {pred_index + 1}"
        
        predictions.append({
            'question': item['question'],
            'prediction': pred,
            'answer': item['answer']
        })
    return predictions