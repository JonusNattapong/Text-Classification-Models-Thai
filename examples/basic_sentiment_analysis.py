"""
Basic Sentiment Analysis Example for Thai Text
A simple implementation of binary sentiment classification using WangchanBERTa.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path to import our pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text_classification_pipeline import TextClassificationPipeline

def create_thai_sentiment_data():
    """Create a larger sample dataset for Thai sentiment analysis."""
    
    positive_samples = [
        "ผมชอบภาพยนตร์เรื่องนี้มาก สนุกและน่าตื่นเต้น",
        "สินค้าคุณภาพดีมาก ราคาไม่แพง คุ้มค่าเงิน แนะนำเลย", 
        "โรงแรมนี้สะอาด สะดวกสบาย ราคาดี วิวสวย",
        "อาหารอร่อยมาก บรรยากาศดี ราคาถูก จะมาอีก",
        "การบริการดีมาก เจ้าหน้าที่ใส่ใจลูกค้า พอใจมาก",
        "คุณภาพเยี่ยม ส่งเร็ว บริการดี ประทับใจมาก",
        "สวยงาม ใช้งานง่าย คุ้มค่ามาก ชอบมาก",
        "ยอดเยี่ยม ทุกอย่างดีมาก จะแนะนำเพื่อน",
        "น่าใช้มาก ดีไซน์สวย คุณภาพดี ราคาเหมาะสม",
        "ประทับใจมาก บริการดีเยี่ยม จะใช้บริการอีก",
        "รสชาติดี หอม อร่อย ราคาเหมาะสม",
        "พนักงานเป็นกันเอง บริการดี สถานที่สะอาด",
        "สินค้าตรงตามคำอธิบาย คุณภาพดี จัดส่งรวดเร็ว",
        "บรรยากาศดีมาก เหมาะกับการพักผ่อน สงบ",
        "เทคโนโลยีทันสมัย ใช้งานสะดวก มีประสิทธิภาพ"
    ]
    
    negative_samples = [
        "ร้านอาหารนี้แย่มาก อาหารไม่อร่อย บริการไม่ดี",
        "บริการแย่มาก พนักงานไม่เป็นมิตร ไม่อยากมาอีก",
        "หนังสือเล่มนี้น่าเบื่อมาก เนื้อหาไม่น่าสนใจ ไม่แนะนำ",
        "สินค้าไม่ตรงตามที่โฆษณา คุณภาพต่ำ เสียเงิน",
        "ของเก่าแล้ว ไม่คุ้มค่า ผิดหวังมาก",
        "ไม่ดีเลย แย่มาก ไม่คุ้มค่าเงิน จะไม่ซื้ออีก",
        "แย่มาก ผิดหวัง ไม่ควรซื้อ เสียเงินเปล่า",
        "ไม่ได้เรื่อง ไม่น่าเชื่อถือ บริการแย่",
        "ไม่ดี คุณภาพต่ำ ไม่คุ้มค่า จะไม่ซื้ออีก",
        "ไม่พอใจ ไม่ตรงตามความต้องการ จะคืนสินค้า",
        "รสชาติแย่ ไม่สด เสียเงิน",
        "พนักงานหยาบคาย บริการไม่ดี ไม่มีมารยาท",
        "สินค้ามีปัญหา ส่งช้า ไม่พอใจ",
        "เสียงดัง รบกวน ไม่สะดวกสบาย",
        "เก่า ช้า ไม่ทันสมัย ใช้งานยาก"
    ]
    
    # Create DataFrame
    data = {
        'text': positive_samples + negative_samples,
        'label': ['positive'] * len(positive_samples) + ['negative'] * len(negative_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """Main function to run sentiment analysis example."""
    
    print("🚀 Thai Sentiment Analysis Example")
    print("=" * 50)
    
    # Create dataset
    print("📊 Creating sample dataset...")
    df = create_thai_sentiment_data()
    
    print(f"   Total samples: {len(df)}")
    print(f"   Positive samples: {len(df[df['label'] == 'positive'])}")
    print(f"   Negative samples: {len(df[df['label'] == 'negative'])}")
    
    # Initialize pipeline
    print("\n🤖 Initializing text classification pipeline...")
    pipeline = TextClassificationPipeline(
        model_name="airesearch/wangchanberta-base-att-spm-uncased",
        num_labels=2,
        max_length=256  # Shorter length for this example
    )
    
    # Load model and tokenizer
    pipeline.load_model_and_tokenizer()
    
    # Prepare dataset
    print("\n📋 Preparing dataset...")
    dataset_dict = pipeline.prepare_dataset(
        df,
        text_column='text',
        label_column='label',
        test_size=0.3,
        val_size=0.2
    )
    
    print(f"   Label mappings: {pipeline.label2id}")
    
    # Configure training (reduced parameters for quick demo)
    training_config = {
        'learning_rate': 2e-5,
        'batch_size': 8,
        'num_epochs': 2,  # Reduced for quick demo
        'warmup_steps': 50,
        'weight_decay': 0.01,
        'eval_steps': 10,
        'logging_steps': 5,
        'save_steps': 50,
        'early_stopping_patience': 2,
        'output_dir': './sentiment_model_results'
    }
    
    # Fine-tune model
    print("\n🎯 Starting fine-tuning...")
    results = pipeline.fine_tune(dataset_dict, **training_config)
    
    print(f"\n✅ Training completed!")
    print(f"   Final test results: {results['test_results']}")
    
    # Test inference
    print("\n🔍 Testing inference...")
    
    test_sentences = [
        "ร้านนี้อาหารอร่อยมาก บริการดี ราคาเหมาะสม",  # Should be positive
        "ไม่พอใจเลย สินค้าแย่ บริการไม่ดี",             # Should be negative  
        "โอเคนะ ใช้ได้ ปกติ",                          # Neutral/positive
        "แย่มากๆ ไม่แนะนำ เสียเงิน"                    # Should be negative
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        prediction = pipeline.predict(sentence)
        probabilities = pipeline.predict(sentence, return_probabilities=True)
        
        print(f"\n{i}. Text: {sentence}")
        print(f"   Prediction: {prediction}")
        print(f"   Probabilities: {dict(zip(pipeline.id2label.values(), probabilities.round(4)))}")
    
    # Evaluate model comprehensively
    print("\n📊 Comprehensive evaluation...")
    evaluation_results = pipeline.evaluate_model(dataset_dict, plot_confusion_matrix=True)
    
    print("\n🎉 Sentiment analysis example completed!")
    print(f"💾 Model saved in: {training_config['output_dir']}")
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()
