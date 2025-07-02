"""
Configuration file for fine-tuning experiments.
Contains various hyperparameter configurations for different scenarios.
"""

# Base configuration for Thai text classification
BASE_CONFIG = {
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

# Quick development configuration (faster training for testing)
QUICK_CONFIG = {
    **BASE_CONFIG,
    'max_length': 256,
    'batch_size': 8,
    'num_epochs': 2,
    'warmup_steps': 50,
    'eval_steps': 50,
    'logging_steps': 5,
    'save_steps': 100,
    'early_stopping_patience': 2
}

# High-performance configuration (for production use)
HIGH_PERFORMANCE_CONFIG = {
    **BASE_CONFIG,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'num_epochs': 5,
    'warmup_steps': 1000,
    'weight_decay': 0.1,
    'gradient_accumulation_steps': 2,
    'early_stopping_patience': 5
}

# Small dataset configuration (for limited data scenarios)
SMALL_DATA_CONFIG = {
    **BASE_CONFIG,
    'learning_rate': 3e-5,
    'batch_size': 8,
    'num_epochs': 5,
    'warmup_steps': 100,
    'weight_decay': 0.05,
    'early_stopping_patience': 3
}

# Multi-class configuration (for topic classification, etc.)
MULTICLASS_CONFIG = {
    **BASE_CONFIG,
    'learning_rate': 1.5e-5,
    'batch_size': 16,
    'num_epochs': 4,
    'warmup_steps': 750,
    'weight_decay': 0.15
}

# Hyperparameter search spaces for optimization
HYPERPARAMETER_SEARCH_SPACES = {
    'basic_search': {
        'learning_rate': (1e-6, 1e-4),
        'batch_size': [8, 16, 32],
        'num_epochs': (2, 5),
        'warmup_ratio': (0.0, 0.2),
        'weight_decay': (0.0, 0.3)
    },
    
    'advanced_search': {
        'learning_rate': (5e-6, 5e-5),
        'batch_size': [8, 16, 24, 32],
        'num_epochs': (2, 6),
        'warmup_ratio': (0.05, 0.25),
        'weight_decay': (0.01, 0.2),
        'gradient_accumulation_steps': [1, 2, 4]
    }
}

# Model options for different use cases
MODEL_OPTIONS = {
    'thai_base': 'airesearch/wangchanberta-base-att-spm-uncased',
    'thai_large': 'airesearch/wangchanberta-large-att-spm-uncased',
    'multilingual_base': 'bert-base-multilingual-cased',
    'xlm_roberta': 'xlm-roberta-base',
    'distil_bert': 'distilbert-base-multilingual-cased'
}

# Dataset configurations
DATASET_CONFIGS = {
    'sentiment_analysis': {
        'text_column': 'text',
        'label_column': 'label',
        'test_size': 0.2,
        'val_size': 0.1,
        'expected_labels': ['positive', 'negative']
    },
    
    'topic_classification': {
        'text_column': 'text',
        'label_column': 'category',
        'test_size': 0.15,
        'val_size': 0.15,
        'expected_labels': ['technology', 'sports', 'politics', 'entertainment', 'business']
    },
    
    'intent_classification': {
        'text_column': 'utterance',
        'label_column': 'intent',
        'test_size': 0.2,
        'val_size': 0.1,
        'expected_labels': ['greeting', 'question', 'request', 'complaint', 'compliment']
    }
}

# Evaluation metrics configuration
EVALUATION_METRICS = {
    'binary_classification': {
        'primary_metric': 'f1',
        'additional_metrics': ['accuracy', 'precision', 'recall', 'auc']
    },
    
    'multiclass_classification': {
        'primary_metric': 'f1_weighted',
        'additional_metrics': ['accuracy', 'f1_macro', 'f1_micro']
    }
}

# Production deployment settings
DEPLOYMENT_CONFIG = {
    'model_save_format': 'pytorch',  # or 'onnx', 'tensorrt'
    'quantization': False,
    'optimization_level': 'O1',  # Mixed precision level
    'max_batch_size': 32,
    'inference_timeout': 30,  # seconds
    'enable_monitoring': True,
    'log_predictions': True
}

def get_config(config_name: str = 'base'):
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'base': BASE_CONFIG,
        'quick': QUICK_CONFIG,
        'high_performance': HIGH_PERFORMANCE_CONFIG,
        'small_data': SMALL_DATA_CONFIG,
        'multiclass': MULTICLASS_CONFIG
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name].copy()

def get_model_name(model_key: str = 'thai_base'):
    """
    Get model name by key.
    
    Args:
        model_key: Key for the model
        
    Returns:
        Model name string
    """
    if model_key not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_OPTIONS.keys())}")
    
    return MODEL_OPTIONS[model_key]

def get_dataset_config(dataset_type: str = 'sentiment_analysis'):
    """
    Get dataset configuration by type.
    
    Args:
        dataset_type: Type of dataset configuration
        
    Returns:
        Dataset configuration dictionary
    """
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_type].copy()

# Usage examples:
# config = get_config('high_performance')
# model_name = get_model_name('thai_large')
# dataset_config = get_dataset_config('topic_classification')
