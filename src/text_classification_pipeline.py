"""
Fine-Tuning Language Models for Text Classification: Core Utilities
A comprehensive toolkit for fine-tuning language models on Thai text classification tasks.
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pythainlp import word_tokenize
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationPipeline:
    """
    A comprehensive pipeline for fine-tuning language models on text classification tasks.
    
    This class provides end-to-end functionality for:
    1. Data preprocessing and tokenization
    2. Model fine-tuning with customizable hyperparameters
    3. Evaluation with comprehensive metrics
    4. Model deployment and inference
    """
    
    def __init__(
        self,
        model_name: str = "airesearch/wangchanberta-base-att-spm-uncased",
        num_labels: int = 2,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the text classification pipeline.
        
        Args:
            model_name: Pre-trained model identifier from Hugging Face
            num_labels: Number of classification labels
            max_length: Maximum sequence length for tokenization
            device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.id2label = {}
        self.label2id = {}
        
        logger.info(f"Initialized pipeline with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {str(e)}")
            raise
    
    def preprocess_text(self, text: str, language: str = 'thai') -> str:
        """
        Preprocess text for Thai language understanding.
        
        Args:
            text: Input text to preprocess
            language: Language of the text ('thai' or 'english')
            
        Returns:
            Preprocessed text
        """
        if language == 'thai':
            # Thai-specific preprocessing
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Optional: Apply Thai word segmentation if needed
            # text = ' '.join(word_tokenize(text, engine='attacut'))
            
        else:
            # English preprocessing
            text = text.lower()
            text = ' '.join(text.split())
        
        return text
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'label',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> DatasetDict:
        """
        Prepare dataset for training with proper train/val/test splits.
        
        Args:
            df: DataFrame containing text and labels
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        # Create label mappings
        unique_labels = df[label_column].unique()
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Convert labels to numeric
        df['labels'] = df[label_column].map(self.label2id)
        
        # Preprocess text
        df['text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['labels']
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['labels']
        )
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        val_dataset = val_dataset.map(self._tokenize_function, batched=True)
        test_dataset = test_dataset.map(self._tokenize_function, batched=True)
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info(f"Dataset prepared:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return dataset_dict
    
    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
            
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        precision, recall, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average='micro'
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune(
        self,
        dataset_dict: DatasetDict,
        output_dir: str = "./results",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "f1_weighted",
        greater_is_better: bool = True,
        early_stopping_patience: int = 3,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        dataloader_num_workers: int = 0,
        **kwargs
    ) -> Dict:
        """
        Fine-tune the model with comprehensive training configuration.
        
        Args:
            dataset_dict: Prepared dataset dictionary
            output_dir: Directory to save model checkpoints
            learning_rate: Learning rate for training
            batch_size: Training batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            logging_steps: Steps between logging
            eval_steps: Steps between evaluations
            save_steps: Steps between model saves
            load_best_model_at_end: Whether to load the best model at the end
            metric_for_best_model: Metric to use for model selection
            greater_is_better: Whether higher metric values are better
            early_stopping_patience: Patience for early stopping
            gradient_accumulation_steps: Steps for gradient accumulation
            fp16: Whether to use mixed precision training
            dataloader_num_workers: Number of dataloader workers
            **kwargs: Additional training arguments
            
        Returns:
            Training history and final metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_strategy="steps",
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=None,  # Disable wandb/tensorboard for now
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16 and torch.cuda.is_available(),
            dataloader_num_workers=dataloader_num_workers,
            **kwargs
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Early stopping callback
        from transformers import EarlyStoppingCallback
        callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate on test set
        test_results = self.trainer.evaluate(dataset_dict['test'])
        
        logger.info("Fine-tuning completed!")
        logger.info(f"Final test results: {test_results}")
        
        return {
            'train_result': train_result,
            'test_results': test_results
        }
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[str, List[str], np.ndarray]:
        """
        Make predictions on new text(s).
        
        Args:
            texts: Single text or list of texts to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (labels or probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                return probabilities[0] if single_input else probabilities
            else:
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                predicted_labels = [self.id2label[pred] for pred in predictions]
                return predicted_labels[0] if single_input else predicted_labels
    
    def evaluate_model(self, dataset_dict: DatasetDict, plot_confusion_matrix: bool = True) -> Dict:
        """
        Comprehensive model evaluation with metrics and visualizations.
        
        Args:
            dataset_dict: Dataset to evaluate on
            plot_confusion_matrix: Whether to plot confusion matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained first.")
        
        # Evaluate on all splits
        results = {}
        
        for split_name, dataset in dataset_dict.items():
            eval_results = self.trainer.evaluate(dataset)
            results[split_name] = eval_results
            
            if plot_confusion_matrix and split_name == 'test':
                # Get predictions for confusion matrix
                predictions = self.trainer.predict(dataset)
                y_pred = np.argmax(predictions.predictions, axis=1)
                y_true = predictions.label_ids
                
                # Plot confusion matrix
                self._plot_confusion_matrix(y_true, y_pred, split_name)
        
        return results
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.id2label.values()),
            yticklabels=list(self.id2label.values())
        )
        plt.title(f'Confusion Matrix - {dataset_name.title()} Set')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_search(
        self,
        dataset_dict: DatasetDict,
        search_space: Dict,
        n_trials: int = 20,
        direction: str = "maximize",
        metric_name: str = "eval_f1_weighted"
    ) -> Dict:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            dataset_dict: Prepared dataset
            search_space: Dictionary defining search space for hyperparameters
            n_trials: Number of optimization trials
            direction: Optimization direction ('maximize' or 'minimize')
            metric_name: Metric to optimize
            
        Returns:
            Best hyperparameters and optimization history
        """
        import optuna
        
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', *search_space.get('learning_rate', [1e-6, 1e-4]))
            batch_size = trial.suggest_categorical('batch_size', search_space.get('batch_size', [8, 16, 32]))
            num_epochs = trial.suggest_int('num_epochs', *search_space.get('num_epochs', [2, 5]))
            warmup_ratio = trial.suggest_float('warmup_ratio', *search_space.get('warmup_ratio', [0.0, 0.2]))
            weight_decay = trial.suggest_float('weight_decay', *search_space.get('weight_decay', [0.0, 0.3]))
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=f"./hp_search_trial_{trial.number}",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model=metric_name.replace('eval_', ''),
                report_to=None,
                disable_tqdm=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                compute_metrics=self.compute_metrics,
            )
            
            # Train and evaluate
            trainer.train()
            eval_results = trainer.evaluate()
            
            return eval_results[metric_name]
        
        # Create study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best {metric_name}: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }

# Additional utility functions

def load_thai_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load sample Thai text classification datasets.
    
    Returns:
        Dictionary of sample datasets
    """
    # Sample Thai sentiment analysis data
    sentiment_data = {
        'text': [
            'ผมชอบภาพยนตร์เรื่องนี้มาก สนุกและน่าตื่นเต้น',
            'ร้านอาหารนี้แย่มาก อาหารไม่อร่อย',
            'สินค้าคุณภาพดี ราคาไม่แพง แนะนำเลย',
            'บริการแย่ พนักงานไม่เป็นมิตร',
            'โรงแรมนี้สะอาด สะดวกสบาย ราคาดี',
            'หนังสือเล่มนี้น่าเบื่อมาก ไม่แนะนำ'
        ],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
    }
    
    # Sample Thai topic classification data
    topic_data = {
        'text': [
            'การแพร่ระบาดของโควิด-19 ส่งผลกระทบต่อเศรษฐกิจโลก',
            'นักฟุตบอลทีมชาติไทยเตรียมความพร้อมสำหรับการแข่งขัน',
            'เทคโนโลยี AI กำลังเปลี่ยนแปลงวิธีการทำงาน',
            'ราคาน้ำมันปรับตัวสูงขึ้นอย่างต่อเนื่อง',
            'การท่องเที่ยวในประเทศไทยเริ่มฟื้นตัว',
            'การพัฒนาแอปพลิเคชันมือถือสำหรับธุรกิจ'
        ],
        'label': ['health', 'sports', 'technology', 'economy', 'travel', 'technology']
    }
    
    return {
        'sentiment': pd.DataFrame(sentiment_data),
        'topic': pd.DataFrame(topic_data)
    }

def create_sample_config() -> Dict:
    """
    Create a sample configuration for fine-tuning.
    
    Returns:
        Sample configuration dictionary
    """
    return {
        'model_name': 'airesearch/wangchanberta-base-att-spm-uncased',
        'max_length': 512,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'num_epochs': 3,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'early_stopping_patience': 3,
        'eval_steps': 500,
        'logging_steps': 10,
        'save_steps': 1000,
        'fp16': True,
        'dataloader_num_workers': 0,
        'gradient_accumulation_steps': 1
    }

# Demo function for quick testing
def demo_pipeline():
    """
    Demonstrate the complete pipeline with sample data.
    """
    print("=== Fine-Tuning Language Models for Text Classification Demo ===\n")
    
    # Load sample datasets
    datasets = load_thai_datasets()
    
    # Initialize pipeline
    pipeline = TextClassificationPipeline(
        model_name="airesearch/wangchanberta-base-att-spm-uncased",
        num_labels=2,
        max_length=256
    )
    
    # Load model and tokenizer
    pipeline.load_model_and_tokenizer()
    
    # Prepare dataset (using sentiment analysis as example)
    dataset_dict = pipeline.prepare_dataset(
        datasets['sentiment'],
        text_column='text',
        label_column='label',
        test_size=0.3,
        val_size=0.2
    )
    
    print("Dataset prepared successfully!")
    print(f"Label mappings: {pipeline.label2id}")
    
    # Fine-tune (with minimal epochs for demo)
    config = create_sample_config()
    config['num_epochs'] = 1  # Reduce for demo
    config['eval_steps'] = 1
    config['logging_steps'] = 1
    
    results = pipeline.fine_tune(dataset_dict, **config)
    
    # Make predictions
    sample_texts = [
        "ผลิตภัณฑ์นี้ดีมาก ฉันแนะนำ",
        "บริการแย่มาก ไม่พอใจ"
    ]
    
    predictions = pipeline.predict(sample_texts)
    probabilities = pipeline.predict(sample_texts, return_probabilities=True)
    
    print("\n=== Predictions ===")
    for text, pred, prob in zip(sample_texts, predictions, probabilities):
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print(f"Probabilities: {prob}")
        print()
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    demo_pipeline()
