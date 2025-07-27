import json
import argparse
from preprocess import load_data
from feature_engineering import extract_features
from model import run_model, run_rule_based, run_zero_shot, fine_tune_model

def evaluate(predictions):
    """Calculate accuracy and print some correct/incorrect cases."""
    correct = 0
    for p in predictions:
        if p['prediction'].strip().lower() == p['answer'].strip().lower():
            correct += 1
    total = len(predictions)
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baselines for QQA dataset.")
    parser.add_argument("--baseline", type=str, choices=["rule-based", "pre-trained", "zero-shot"], required=True,
                        help="Choose which baseline to run: rule-based, pre-trained, or zero-shot.")
    args = parser.parse_args()

    dataset_names = ['dev', 'test', 'train']
    
    if args.baseline == "pre-trained":
        # Fine-tune the model on the train dataset first
        train_path = '/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_train_processed.json'
        train_data = load_data(train_path)
        print("Fine-tuning the model on the train dataset...")
        fine_tune_model(train_data)
        print("Fine-tuning completed.")

        # Process dev and test datasets using the fine-tuned model
        for name in ['dev', 'test']:
            input_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}_processed.json'
            output_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/results/output_{name}_{args.baseline}.json'
            print(f"Processing {name} dataset using {args.baseline} baseline...")
            data = load_data(input_path)
            data = extract_features(data)
            predictions = run_model(data, use_fine_tuned=True)
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions for {name} dataset to {output_path}")
            accuracy = evaluate(predictions)
            print(f"{name.capitalize()} Accuracy ({args.baseline}): {accuracy:.2f}")
    else:
        # For rule-based and zero-shot, process all datasets
        for name in dataset_names:
            input_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/data/QQA_{name}_processed.json'
            output_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/results/output_{name}_{args.baseline}.json'
            print(f"Processing {name} dataset using {args.baseline} baseline...")
            data = load_data(input_path)
            if args.baseline == "rule-based":
                predictions = run_rule_based(data)
            elif args.baseline == "zero-shot":
                predictions = run_zero_shot(data)
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions for {name} dataset to {output_path}")
            accuracy = evaluate(predictions)
            print(f"{name.capitalize()} Accuracy ({args.baseline}): {accuracy:.2f}")
            
    metrics_path = f'/Users/tikam/Documents/NLP/PROJECT/baseline/results/evaluation_metrics_{args.baseline}.txt'
    with open(metrics_path, 'w') as f:
        if args.baseline == "pre-trained":
            for name in ['dev', 'test']:
                predictions = load_data(f'/Users/tikam/Documents/NLP/PROJECT/baseline/results/output_{name}_{args.baseline}.json')
                accuracy = evaluate(predictions)
                f.write(f"{name.capitalize()} Accuracy ({args.baseline}): {accuracy:.2f}\n")
        else:
            for name in dataset_names:
                predictions = load_data(f'/Users/tikam/Documents/NLP/PROJECT/baseline/results/output_{name}_{args.baseline}.json')
                accuracy = evaluate(predictions)
                f.write(f"{name.capitalize()} Accuracy ({args.baseline}): {accuracy:.2f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")