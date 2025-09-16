"""
Recommendation dataset for semantic ID experiments.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
from datasets import load_dataset

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation tasks with semantic IDs.
    
    Based on the experiment in omni_qwen.ipynb.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        item2index: Optional[Dict] = None,
        max_history_length: int = 10,
        min_history_length: int = 2
    ):
        """
        Initialize recommendation dataset.
        
        Args:
            df: DataFrame with columns ['user_id', 'nm_ids', 'titles']
            item2index: Mapping from item names to indices
            max_history_length: Maximum history length to consider
            min_history_length: Minimum history length required
        """
        self.df = df
        self.item2index = item2index or {}
        self.max_history_length = max_history_length
        self.min_history_length = min_history_length
        
        # Process the data
        self.examples = self._prepare_examples()
        logger.info(f"Created {len(self.examples)} recommendation examples")
    
    def _prepare_examples(self):
        """Prepare examples from the dataframe."""
        examples = []
        
        for row in self.df.itertuples(index=False):
            user_id = row.user_id
            ids = row.nm_ids if hasattr(row, 'nm_ids') else []
            titles = row.titles if hasattr(row, 'titles') else []
            
            # Filter valid IDs
            if self.item2index:
                valid_ids = [
                    self.item2index[nm_id] 
                    for nm_id in ids 
                    if nm_id in self.item2index
                ]
            else:
                valid_ids = ids
            
            # Skip if not enough history
            if len(valid_ids) < self.min_history_length:
                continue
            
            # Truncate if too long
            if len(valid_ids) > self.max_history_length:
                valid_ids = valid_ids[-self.max_history_length:]
                titles = titles[-self.max_history_length:]
            
            # Create examples for each position in history
            for i in range(1, len(valid_ids)):
                history_ids = valid_ids[:i]
                history_titles = titles[:i]
                target_id = valid_ids[i]
                
                # Create conversation format
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are a recommendation system assistant designed to provide "
                                    "personalized product suggestions based on user purchase history "
                                    "and preferences. You can process both textual information and "
                                    "item IDs representing products the user interacted with. "
                                    "Respond clearly and helpfully."
                                )
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"user {user_id}, history purchase: " + ", ".join(history_titles)
                            }
                        ]
                    },
                ]
                
                examples.append({
                    "conversation": conversation,
                    "history_ids": history_ids,
                    "target_id": target_id,
                    "user_id": user_id
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            example["conversation"],
            example["history_ids"],
            example["target_id"]
        )


class RecommendationCollator:
    """
    Collator for recommendation dataset.
    
    Handles batching of conversations, ID sequences, and targets.
    """
    
    def __init__(self, tokenizer=None, device='cpu'):
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, batch):
        """
        Collate a batch of recommendation examples.
        
        Args:
            batch: List of (conversation, history_ids, target_id) tuples
            
        Returns:
            Dictionary with batched data
        """
        conversations, id_sequences, targets = zip(*batch)
        
        # Extract user texts from conversations
        texts = []
        for conv in conversations:
            # Find user message
            user_msg = None
            for msg in conv:
                if msg["role"] == "user":
                    user_msg = msg
                    break
            
            if user_msg and "content" in user_msg:
                # Extract text from content
                for content_item in user_msg["content"]:
                    if content_item["type"] == "text":
                        texts.append(content_item["text"])
                        break
                else:
                    texts.append("")
            else:
                texts.append("")
        
        # Convert targets to tensor (keep on CPU for now)
        labels = torch.tensor(targets, dtype=torch.long)
        
        # Convert id_sequences to tensors (keep on CPU for now)
        id_tensors = []
        for ids in id_sequences:
            id_tensor = torch.tensor(ids, dtype=torch.long)
            id_tensors.append(id_tensor)
        
        return {
            "text": texts,
            "id_ids": id_tensors,
            "labels": labels
        }


def load_item_mapping(mapping_path: str) -> Dict:
    """
    Load item name to index mapping.
    
    Args:
        mapping_path: Path to pickle file with mapping
        
    Returns:
        Dictionary mapping item names to indices
    """
    try:
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        logger.info(f"Loaded item mapping with {len(mapping)} items")
        return mapping
    except Exception as e:
        logger.warning(f"Failed to load item mapping from {mapping_path}: {e}")
        return {}


def prepare_recommendation_data(
    dataset_name: str = "seniichev/amazon-fashion-2023-full",
    user_id_field: str = "user_id",
    item_id_field: str = "parent_asin", 
    title_field: str = "title",
    max_history_length: int = 10,
    min_history_length: int = 2,
    random_seed: int = 42
) -> tuple:
    """
    Prepare recommendation dataset from Hugging Face dataset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        user_id_field: Field name for user IDs
        item_id_field: Field name for item IDs
        title_field: Field name for item titles
        max_history_length: Maximum history length to consider
        min_history_length: Minimum history length required
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df, item2index)
    """
    logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Use the default split (usually 'train')
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
    
    logger.info(f"Loaded {len(df)} records from {dataset_name}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Create item mapping from unique items
    unique_items = df[item_id_field].unique()
    item2index = {item: idx for idx, item in enumerate(unique_items)}
    logger.info(f"Created item mapping with {len(item2index)} unique items")
    
    # Group by user to create user histories
    logger.info("Creating user histories...")
    user_histories = []
    
    for user_id, user_data in df.groupby(user_id_field):
        # Sort by timestamp if available, otherwise by index
        if 'timestamp' in user_data.columns:
            user_data = user_data.sort_values('timestamp')
        else:
            user_data = user_data.sort_index()
        
        # Extract item IDs and titles
        item_ids = user_data[item_id_field].tolist()
        titles = user_data[title_field].tolist()
        
        # Filter out None/NaN values
        valid_items = []
        valid_titles = []
        for item_id, title in zip(item_ids, titles):
            if pd.notna(item_id) and pd.notna(title) and str(item_id).strip() != '':
                valid_items.append(item_id)
                valid_titles.append(str(title).strip())
        
        # Skip users with insufficient history
        if len(valid_items) < min_history_length:
            continue
            
        # Truncate if too long
        if len(valid_items) > max_history_length:
            valid_items = valid_items[-max_history_length:]
            valid_titles = valid_titles[-max_history_length:]
        
        user_histories.append({
            'user_id': user_id,
            'nm_ids': valid_items,
            'titles': valid_titles
        })
    
    # Convert to DataFrame
    df_histories = pd.DataFrame(user_histories)
    logger.info(f"Created {len(df_histories)} user histories")
    
    # Split data
    df_histories = df_histories.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    train_size = int(len(df_histories) * 0.8)
    val_size = int(len(df_histories) * 0.1)
    
    train_df = df_histories[:train_size]
    val_df = df_histories[train_size:train_size + val_size]
    test_df = df_histories[train_size + val_size:]
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df, item2index
