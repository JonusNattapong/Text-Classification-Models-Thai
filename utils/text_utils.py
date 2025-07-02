"""
Utility functions for text preprocessing, evaluation, and data handling.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from collections import Counter

def clean_thai_text(text: str) -> str:
    """
    Clean Thai text by removing unwanted characters and normalizing.
    
    Args:
        text: Input Thai text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep Thai characters, numbers, and basic punctuation
    text = re.sub(r'[^\u0E00-\u0E7F\w\s.,!?()-]', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return text.strip()

def augment_thai_text(text: str, augmentation_type: str = 'synonym') -> List[str]:
    """
    Simple text augmentation for Thai text.
    
    Args:
        text: Input text
        augmentation_type: Type of augmentation ('synonym', 'insertion', 'deletion')
        
    Returns:
        List of augmented texts
    """
    augmented_texts = [text]  # Include original
    
    if augmentation_type == 'synonym':
        # Simple synonym replacement (expand this with actual Thai synonyms)
        synonym_map = {
            'ดี': ['เยี่ยม', 'ยอดเยี่ยม', 'ดีเลิศ'],
            'แย่': ['ไม่ดี', 'ย่ำแย่', 'ไม่เยี่ยม'],
            'สวย': ['งาม', 'สวยงาม', 'ดูดี'],
            'อร่อย': ['รสชาติดี', 'น่าทาน', 'รสเด็ด']
        }
        
        for original, synonyms in synonym_map.items():
            if original in text:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms
                    augmented_text = text.replace(original, synonym)
                    augmented_texts.append(augmented_text)
    
    return augmented_texts

def analyze_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> Dict[str, Any]:
    """
    Analyze text statistics for a DataFrame.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the text column
        
    Returns:
        Dictionary of statistics
    """
    texts = df[text_column].astype(str)
    
    # Basic statistics
    char_counts = texts.str.len()
    word_counts = texts.str.split().str.len()
    
    stats = {
        'total_texts': len(texts),
        'avg_char_length': char_counts.mean(),
        'median_char_length': char_counts.median(),
        'max_char_length': char_counts.max(),
        'min_char_length': char_counts.min(),
        'avg_word_count': word_counts.mean(),
        'median_word_count': word_counts.median(),
        'max_word_count': word_counts.max(),
        'min_word_count': word_counts.min(),
        'std_char_length': char_counts.std(),
        'std_word_count': word_counts.std()
    }
    
    return stats

def plot_text_length_distribution(df: pd.DataFrame, text_column: str = 'text', figsize: Tuple[int, int] = (12, 5)):
    """
    Plot text length distribution.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the text column
        figsize: Figure size tuple
    """
    texts = df[text_column].astype(str)
    char_counts = texts.str.len()
    word_counts = texts.str.split().str.len()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Character length distribution
    ax1.hist(char_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Character Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Character Length Distribution')
    ax1.axvline(char_counts.mean(), color='red', linestyle='--', label=f'Mean: {char_counts.mean():.1f}')
    ax1.legend()
    
    # Word count distribution
    ax2.hist(word_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Word Count Distribution')
    ax2.axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.1f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def balance_dataset(df: pd.DataFrame, label_column: str = 'label', method: str = 'oversample') -> pd.DataFrame:
    """
    Balance dataset using oversampling or undersampling.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        method: 'oversample' or 'undersample'
        
    Returns:
        Balanced DataFrame
    """
    label_counts = df[label_column].value_counts()
    
    if method == 'oversample':
        # Oversample to match the largest class
        target_size = label_counts.max()
        balanced_dfs = []
        
        for label, count in label_counts.items():
            label_df = df[df[label_column] == label]
            if count < target_size:
                # Oversample
                n_samples = target_size - count
                oversampled = label_df.sample(n=n_samples, replace=True, random_state=42)
                balanced_dfs.append(pd.concat([label_df, oversampled]))
            else:
                balanced_dfs.append(label_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    elif method == 'undersample':
        # Undersample to match the smallest class
        target_size = label_counts.min()
        balanced_dfs = []
        
        for label in label_counts.index:
            label_df = df[df[label_column] == label]
            sampled_df = label_df.sample(n=target_size, random_state=42)
            balanced_dfs.append(sampled_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    else:
        raise ValueError("Method must be 'oversample' or 'undersample'")

def create_advanced_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], 
                                   normalize: str = 'true', figsize: Tuple[int, int] = (10, 8)):
    """
    Create an advanced confusion matrix with additional statistics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        normalize: Normalization method ('true', 'pred', 'all', or None)
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.3f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title(f'Confusion Matrix ({normalize} normalized)' if normalize else 'Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

def compute_class_weights(labels: List[str]) -> Dict[str, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary of class weights
    """
    label_counts = Counter(labels)
    total_samples = len(labels)
    n_classes = len(label_counts)
    
    class_weights = {}
    for label, count in label_counts.items():
        weight = total_samples / (n_classes * count)
        class_weights[label] = weight
    
    return class_weights

def save_predictions_with_confidence(texts: List[str], true_labels: List[str], 
                                   predicted_labels: List[str], confidences: List[float],
                                   filepath: str = 'predictions_analysis.csv'):
    """
    Save predictions with confidence scores for further analysis.
    
    Args:
        texts: Input texts
        true_labels: True labels
        predicted_labels: Predicted labels
        confidences: Prediction confidences
        filepath: Output file path
    """
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'confidence': confidences,
        'correct': np.array(true_labels) == np.array(predicted_labels),
        'text_length': [len(text) for text in texts],
        'word_count': [len(text.split()) for text in texts]
    })
    
    # Add confidence category
    results_df['confidence_category'] = pd.cut(
        results_df['confidence'], 
        bins=[0, 0.6, 0.8, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    results_df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")
    
    return results_df

def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                        train_metrics: List[float], val_metrics: List[float],
                        metric_name: str = 'F1 Score'):
    """
    Plot training and validation learning curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_metrics: Training metrics
        val_metrics: Validation metrics
        metric_name: Name of the metric being plotted
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metric curves
    ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_data_summary_report(df: pd.DataFrame, text_column: str = 'text', 
                             label_column: str = 'label') -> str:
    """
    Create a comprehensive data summary report.
    
    Args:
        df: DataFrame to analyze
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Formatted report string
    """
    stats = analyze_text_statistics(df, text_column)
    label_dist = df[label_column].value_counts()
    
    report = f"""
📊 DATA SUMMARY REPORT
{'='*50}

📋 Basic Information:
   • Total samples: {stats['total_texts']:,}
   • Number of classes: {len(label_dist)}
   • Classes: {list(label_dist.index)}

📏 Text Length Statistics:
   • Average character length: {stats['avg_char_length']:.1f}
   • Average word count: {stats['avg_word_count']:.1f}
   • Character length range: {stats['min_char_length']:.0f} - {stats['max_char_length']:.0f}
   • Word count range: {stats['min_word_count']:.0f} - {stats['max_word_count']:.0f}

📈 Class Distribution:
"""
    
    for label, count in label_dist.items():
        percentage = (count / len(df)) * 100
        report += f"   • {label}: {count:,} samples ({percentage:.1f}%)\n"
    
    # Check for class imbalance
    min_class_pct = (label_dist.min() / len(df)) * 100
    max_class_pct = (label_dist.max() / len(df)) * 100
    imbalance_ratio = max_class_pct / min_class_pct
    
    report += f"\n⚖️  Class Balance Analysis:\n"
    report += f"   • Imbalance ratio: {imbalance_ratio:.2f}:1\n"
    
    if imbalance_ratio > 2:
        report += "   • ⚠️ Significant class imbalance detected\n"
        report += "   • Consider using class weights or resampling techniques\n"
    else:
        report += "   • ✅ Classes are reasonably balanced\n"
    
    return report

# Example usage functions
def demonstrate_utilities():
    """Demonstrate utility functions with sample data."""
    
    # Create sample data
    sample_data = {
        'text': [
            'ผมชอบภาพยนตร์เรื่องนี้มาก',
            'ร้านอาหารนี้แย่มาก',
            'สินค้าคุณภาพดี',
            'บริการไม่ดี',
            'ยอดเยี่ยมมาก'
        ],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("🔧 Demonstrating utility functions:")
    print("=" * 40)
    
    # Text statistics
    stats = analyze_text_statistics(df)
    print(f"📊 Text statistics: {stats}")
    
    # Data summary report
    report = create_data_summary_report(df)
    print(report)
    
    # Text augmentation
    augmented = augment_thai_text("ดีมาก", "synonym")
    print(f"🔄 Augmented text: {augmented}")

if __name__ == "__main__":
    demonstrate_utilities()
