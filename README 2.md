# QQA Baseline Models

This project implements three baseline models for the QQA dataset:
1. **Rule-based Baseline**: A simple system that extracts numerical values and directly compares them without contextual understanding.
2. **Pre-trained Model Baseline**: A fine-tuned RoBERTa model on the QQA dataset.
3. **Zero-shot Learning Baseline**: Using a pre-trained model (`facebook/bart-large-mnli`) without fine-tuning.


## Running the Code
### Rule-based Baseline
python src/main.py --baseline rule-based


### Fine-tuned RoBERTa Model
Pre-trained Model Baseline
To run the pre-trained model baseline:
Ensure you have a fine-tuned RoBERTa model saved locally. Update the path in model.py:
model = AutoModelForSequenceClassification.from_pretrained("/path/to/fine-tuned-roberta")

python src/main.py --baseline pre-trained


### Zero-shot Approach
python src/main.py --baseline zero-shot


## Results
The baseline results are stored in the `results/` directory. The files contain accuracy metrics and confusion matrices for each approach.

## Outputs
Predictions:
Predictions for each dataset (dev, test, train) are saved in the results/ directory.
Example file: results/output_dev_rule-based.json
Evaluation Metrics:

Accuracy for each dataset is saved in a text file in the results/ directory.
Example file: results/evaluation_metrics_rule-based.txt
Sample Output in Terminal:

For each dataset, the script prints the accuracy and a few sample predictions:

Processing dev dataset using rule-based baseline...
Saved predictions for dev dataset to results/output_dev_rule-based.json
Dev Accuracy (rule-based): 0.85

Sample Predictions:
Question 1: What is the distance if speed is 10 and time is 2?
Prediction: 'Option 1', Answer: 'Option 1'
--------------------------------------------------

## Note


## Contact
For questions, contact workwithtikam@gmail.com