#!/usr/bin/env python3
"""
Multi-task dataset preparation script for Amazon Fashion.
Supports configurable dataset size and parallel processing.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FashionMultiTaskPreparator:
    """Prepares multi-task dataset for Amazon Fashion."""
    
    def __init__(self, output_dir: str = "fashion_multitask_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "conversations").mkdir(exist_ok=True)
        
        # Load full dataset once
        logger.info("Loading full Amazon Fashion dataset...")
        self.full_dataset = load_dataset('seniichev/amazon-fashion-2023-full', split='train')
        logger.info(f"Loaded {len(self.full_dataset)} products")
    
    def create_product_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Creates dialogue for product analysis."""
        title = str(item.get('title', ''))
        features = item.get('features', [])
        rating = float(item.get('rating', 0))
        
        if isinstance(features, list):
            features_text = ", ".join([str(f) for f in features[:5]])
        else:
            features_text = str(features)
        
        # Determine quality based on rating
        if rating >= 4.5:
            quality_level = "excellent"
        elif rating >= 4.0:
            quality_level = "good"
        elif rating >= 3.5:
            quality_level = "satisfactory"
        else:
            quality_level = "poor"
        
        conversation = {
            "task_type": "product_analysis",
            "conversations": [
                {
                    "role": "user",
                    "content": f"Analyze this product: {title}. What are its features and how good is it?"
                },
                {
                    "role": "assistant",
                    "content": f"Analyzing product '{title}'.\n\n**Features:** {features_text}\n\n**Quality:** {quality_level} (rating: {rating}/5.0)\n\n**Recommendation:** {'I recommend buying' if rating >= 4.0 else 'Consider alternatives' if rating < 3.5 else 'Might be suitable, but check reviews'}."
                }
            ]
        }
        return conversation
    
    def create_next_purchase_recommendation(self, item: Dict[str, Any], all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates dialogue for next purchase recommendations."""
        title = str(item.get('title', ''))
        category = self._extract_category(title)
        
        # Find similar products in the same category
        similar_items = []
        for other_item in all_items[:1000]:  # Limit search for performance
            if other_item != item:
                other_category = self._extract_category(str(other_item.get('title', '')))
                if other_category == category:
                    similar_items.append(other_item)
        
        if len(similar_items) >= 3:
            recommendations = random.sample(similar_items, min(3, len(similar_items)))
            rec_text = "\n".join([f"- {str(rec.get('title', ''))} (rating: {rec.get('rating', 'N/A')})" for rec in recommendations])
        else:
            rec_text = "Unfortunately, no similar products found in this category."
        
        conversation = {
            "task_type": "next_purchase_recommendation",
            "conversations": [
                {
                    "role": "user",
                    "content": f"I bought {title}. What else can I buy in the same category?"
                },
                {
                    "role": "assistant",
                    "content": f"Great choice! Here's what else you can consider in the '{category}' category:\n\n{rec_text}\n\nThese products will complement your wardrobe and create stylish combinations."
                }
            ]
        }
        return conversation
    
    def create_product_comparison(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Dict[str, Any]:
        """Creates dialogue for comparing two products."""
        title1 = str(item1.get('title', ''))
        title2 = str(item2.get('title', ''))
        rating1 = float(item1.get('rating', 0))
        rating2 = float(item2.get('rating', 0))
        
        if rating1 > rating2:
            winner = title1
            reason = f"higher rating ({rating1} vs {rating2})"
        elif rating2 > rating1:
            winner = title2
            reason = f"higher rating ({rating2} vs {rating1})"
        else:
            winner = "both products"
            reason = "same ratings"
        
        conversation = {
            "task_type": "product_comparison",
            "conversations": [
                {
                    "role": "user",
                    "content": f"Compare these two products: 1) {title1} and 2) {title2}. Which one is better?"
                },
                {
                    "role": "assistant",
                    "content": f"Comparing products:\n\n**Product 1:** {title1} (rating: {rating1}/5)\n**Product 2:** {title2} (rating: {rating2}/5)\n\n**Conclusion:** {winner} show better results because they have {reason}.\n\nI recommend choosing the product with higher rating for better quality."
                }
            ]
        }
        return conversation
    
    def create_personalized_recommendation(self, user_history: List[Dict[str, Any]], target_item: Dict[str, Any]) -> Dict[str, Any]:
        """Creates dialogue for personalized recommendations."""
        history_titles = [str(item.get('title', '')) for item in user_history]
        target_title = str(target_item.get('title', ''))
        
        conversation = {
            "task_type": "personalized_recommendation",
            "conversations": [
                {
                    "role": "user",
                    "content": f"I have: {', '.join(history_titles)}. What else would you recommend?"
                },
                {
                    "role": "assistant",
                    "content": f"Analyzing your style based on purchases: {', '.join(history_titles)}.\n\n**Recommendation:** {target_title}\n\nThis product will perfectly complement your wardrobe and create new stylish combinations with your existing items."
                }
            ]
        }
        return conversation
    
    def create_review_generation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Creates dialogue for review generation."""
        title = str(item.get('title', ''))
        rating = float(item.get('rating', 0))
        
        if rating >= 4.5:
            sentiment = "enthusiastic"
            review_text = "Exceeded all my expectations! Quality is outstanding, material feels great, fit is perfect."
        elif rating >= 4.0:
            sentiment = "positive"
            review_text = "Very satisfied with the purchase. Product matches description, quality is good."
        elif rating >= 3.5:
            sentiment = "neutral"
            review_text = "Product is generally decent, but has some minor flaws. Price-quality ratio is acceptable."
        else:
            sentiment = "critical"
            review_text = "Not recommended. Quality leaves much to be desired, expectations were not met."
        
        conversation = {
            "task_type": "review_generation",
            "conversations": [
                {
                    "role": "user",
                    "content": f"Write a {sentiment} review for product: {title}"
                },
                {
                    "role": "assistant",
                    "content": f"Here's my {sentiment} review for product '{title}':\n\n{review_text}\n\nRating: {rating}/5 stars\n\nRecommend: {'yes' if rating >= 4.0 else 'no'}."
                }
            ]
        }
        return conversation
    
    def _extract_category(self, title: str) -> str:
        """Extracts product category from title."""
        title_lower = title.lower()
        if any(word in title_lower for word in ['dress', 'платье']):
            return "dresses"
        elif any(word in title_lower for word in ['pants', 'trousers', 'брюки']):
            return "pants"
        elif any(word in title_lower for word in ['shirt', 'blouse', 'рубашка']):
            return "shirts"
        elif any(word in title_lower for word in ['skirt', 'юбка']):
            return "skirts"
        elif any(word in title_lower for word in ['jacket', 'coat', 'куртка']):
            return "outerwear"
        elif any(word in title_lower for word in ['shoes', 'boots', 'sneakers', 'обувь']):
            return "footwear"
        elif any(word in title_lower for word in ['bag', 'purse', 'сумка']):
            return "accessories"
        else:
            return "other"
    
    def process_item(self, item: Dict[str, Any], item_idx: int, all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes one product and creates dialogues for all task types."""
        conversations = []
        
        # 1. Product analysis
        product_conv = self.create_product_analysis(item)
        conversations.append(product_conv)
        
        # 2. Next purchase recommendation
        next_purchase_conv = self.create_next_purchase_recommendation(item, all_items)
        conversations.append(next_purchase_conv)
        
        # 3. Product comparison (if available)
        if item_idx < len(all_items) - 1:
            item2 = all_items[item_idx + 1]
            comparison_conv = self.create_product_comparison(item, item2)
            conversations.append(comparison_conv)
        
        # 4. Personalized recommendation (simulate user history)
        available_items = [i for i in all_items[:1000] if i != item]
        if len(available_items) >= 3:
            user_history = random.sample(available_items, min(3, len(available_items)))
            personalized_conv = self.create_personalized_recommendation(user_history, item)
            conversations.append(personalized_conv)
        
        # 5. Review generation
        review_conv = self.create_review_generation(item)
        conversations.append(review_conv)
        
        # Add metadata to all conversations
        for conv in conversations:
            conv["metadata"] = {
                "asin": str(item.get('parent_asin', '')),
                "rating": float(item.get('rating', 0)),
                "title": str(item.get('title', '')),
                "image_paths": [],
                "task_type": conv["task_type"]
            }
        
        return conversations
    
    def process_chunk(self, chunk_items: List[Dict[str, Any]], chunk_id: int, total_chunks: int) -> List[Dict[str, Any]]:
        """Processes a chunk of products."""
        logger.info(f"Processing chunk {chunk_id + 1}/{total_chunks} ({len(chunk_items)} products)")
        
        all_conversations = []
        for i, item in enumerate(chunk_items):
            try:
                conversations = self.process_item(item, i, chunk_items)
                all_conversations.extend(conversations)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Chunk {chunk_id + 1}: processed {i + 1}/{len(chunk_items)} products")
                    
            except Exception as e:
                logger.warning(f"Error processing product in chunk {chunk_id + 1}: {e}")
                continue
        
        logger.info(f"Chunk {chunk_id + 1} ready: {len(all_conversations)} dialogues")
        return all_conversations
    
    def prepare_dataset(self, max_items: int = None, chunk_size: int = 10000, use_multiprocessing: bool = True):
        """Prepares multi-task dataset."""
        if max_items is None:
            max_items = len(self.full_dataset)
        
        logger.info(f"Preparing dataset (max {max_items} products, chunk: {chunk_size})")
        
        # Limit number of products
        items = list(self.full_dataset.select(range(min(max_items, len(self.full_dataset)))))
        logger.info(f"Loaded {len(items)} products")
        
        if use_multiprocessing and len(items) > chunk_size:
            # Split into chunks for parallel processing
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            logger.info(f"Split into {len(chunks)} chunks of {chunk_size} products")
            
            # Parallel processing
            with mp.Pool(processes=min(mp.cpu_count(), len(chunks))) as pool:
                process_func = partial(self.process_chunk, chunk_id=0, total_chunks=len(chunks))
                chunk_results = pool.map(process_func, chunks)
            
            all_conversations = []
            for chunk_convs in chunk_results:
                all_conversations.extend(chunk_convs)
        else:
            # Sequential processing
            all_conversations = self.process_chunk(items, 0, 1)
        
        logger.info(f"Created {len(all_conversations)} dialogues for {len(items)} products")
        
        # Split into train/validation
        split_index = int(len(all_conversations) * 0.8)
        train_conversations = all_conversations[:split_index]
        val_conversations = all_conversations[split_index:]
        
        # Save
        train_path = self.output_dir / "conversations" / "train.jsonl"
        val_path = self.output_dir / "conversations" / "validation.jsonl"
        
        logger.info("Saving train set...")
        with open(train_path, 'w', encoding='utf-8') as f:
            for conv in tqdm(train_conversations, desc="Train"):
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        logger.info("Saving validation set...")
        with open(val_path, 'w', encoding='utf-8') as f:
            for conv in tqdm(val_conversations, desc="Validation"):
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        # Task type statistics
        task_counts = {}
        for conv in all_conversations:
            task_type = conv["task_type"]
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        logger.info("Task type distribution:")
        for task_type, count in task_counts.items():
            logger.info(f"  {task_type}: {count}")
        
        logger.info(f"Multi-task dataset saved to {self.output_dir}")
        logger.info(f"Train: {train_path} ({len(train_conversations)} dialogues)")
        logger.info(f"Validation: {val_path} ({len(val_conversations)} dialogues)")
        
        return self.output_dir

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare multi-task dataset for Amazon Fashion")
    parser.add_argument("--max_items", type=int, default=None, 
                       help="Maximum number of products (default: all)")
    parser.add_argument("--chunk_size", type=int, default=10000,
                       help="Chunk size for parallel processing")
    parser.add_argument("--no_multiprocessing", action="store_true",
                       help="Disable multiprocessing")
    parser.add_argument("--output_dir", type=str, default="fashion_multitask_dataset",
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    preparator = FashionMultiTaskPreparator(output_dir=args.output_dir)
    output_dir = preparator.prepare_dataset(
        max_items=args.max_items,
        chunk_size=args.chunk_size,
        use_multiprocessing=not args.no_multiprocessing
    )
    
    print(f"\nMulti-task dataset ready! Located at: {output_dir}")
    print("Now you can train omnimodal LLM on multiple tasks!")

if __name__ == "__main__":
    main()
