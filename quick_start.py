"""
Quick Start Example for Thai Text Classification
Run this script to see the pipeline in action with minimal setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_classification_pipeline import TextClassificationPipeline
import pandas as pd

def quick_demo():
    """Quick demonstration of the text classification pipeline."""
    
    print("🚀 Thai Text Classification - Quick Start Demo")
    print("=" * 60)
    
    # Sample data
    data = {
        'text': [
            'สินค้าดีมาก คุณภาพเยี่ยม ประทับใจมาก แนะนำเลย',
            'ไม่พอใจเลย บริการแย่มาก ไม่คุ้มค่าเงิน',
            'โรงแรมสะอาด พนักงานเป็นมิตร ราคาเหมาะสม',
            'อาหารไม่อร่อย เสียเงิน ไม่ได้มาตรฐาน',
            'ยอดเยี่ยมมาก บริการดี จะใช้บริการอีก'
        ],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Sample Data:")
    for i, row in df.iterrows():
        print(f"   {i+1}. Text: {row['text']}")
        print(f"      Label: {row['label']}")
    
    print(f"\n🤖 Initializing Pipeline...")
    
    # Initialize with a smaller model for quick demo
    pipeline = TextClassificationPipeline(
        model_name="airesearch/wangchanberta-base-att-spm-uncased",
        num_labels=2,
        max_length=128  # Shorter for quick processing
    )
    
    # Load model
    print("📥 Loading model and tokenizer...")
    pipeline.load_model_and_tokenizer()
    
    # Prepare dataset
    print("📋 Preparing dataset...")
    dataset_dict = pipeline.prepare_dataset(
        df,
        text_column='text',
        label_column='label',
        test_size=0.2,
        val_size=0.2
    )
    
    print(f"   Label mappings: {pipeline.label2id}")
    
    # Quick training (minimal epochs)
    print("\n🎯 Quick training (1 epoch for demo)...")
    
    training_config = {
        'learning_rate': 3e-5,
        'batch_size': 2,
        'num_epochs': 1,
        'warmup_steps': 10,
        'weight_decay': 0.01,
        'eval_steps': 2,
        'logging_steps': 1,
        'save_steps': 10,
        'early_stopping_patience': 1,
        'output_dir': './quick_demo_results'
    }
    
    results = pipeline.fine_tune(dataset_dict, **training_config)
    
    print(f"\n✅ Training completed!")
    
    # Test predictions
    print(f"\n🔍 Testing predictions:")
    
    test_texts = [
        "ร้านนี้อาหารอร่อยมาก บรรยากาศดี",  # Should be positive
        "ไม่ดีเลย แย่มาก ไม่พอใจ"             # Should be negative
    ]
    
    for i, text in enumerate(test_texts, 1):
        prediction = pipeline.predict(text)
        probabilities = pipeline.predict(text, return_probabilities=True)
        
        print(f"\n   {i}. Text: {text}")
        print(f"      Prediction: {prediction}")
        print(f"      Confidence: {max(probabilities):.4f}")
        
        # Show all probabilities
        prob_dict = dict(zip(pipeline.id2label.values(), probabilities))
        print(f"      All probabilities: {prob_dict}")
    
    print(f"\n🎉 Quick demo completed!")
    print(f"💾 Model saved in: {training_config['output_dir']}")
    print(f"\n📚 Next steps:")
    print("   1. Try the full notebook: notebooks/fine_tuning_guide.ipynb")
    print("   2. Run examples: python examples/basic_sentiment_analysis.py")
    print("   3. Deploy with API: python deployment/api_server.py")
    
    return pipeline

if __name__ == "__main__":
    try:
        pipeline = quick_demo()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have installed all requirements: pip install -r requirements.txt")
        print("   2. Check if you have sufficient GPU/CPU memory")
        print("   3. Try reducing batch_size if you encounter memory issues")
        sys.exit(1)
