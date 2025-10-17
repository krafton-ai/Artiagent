import json
from typing import List, Dict, Any
from .vqa_builders import QAPair

class VQASerializer:
    """Serialize Q-A pairs to LLaMA-Factory / Qwen2.5-VL format."""
    
    @staticmethod
    def serialize_conversation(
        images: List[str],
        qa_pairs: List[QAPair]
    ) -> Dict[str, Any]:
        """Serialize a conversation to JSON format.
        
        Adds <image> token(s) to the first question only:
        - Single image: <image>\n
        - Two images: <image> <image>\n
        
        Args:
            images: List of image paths
            qa_pairs: List of Q-A pairs
        
        Returns:
            Dictionary in LLaMA-Factory format
        """
        conversations = []
        num_images = len(images)
        
        for i, qa in enumerate(qa_pairs):
            question = qa.question
            
            # Add <image> token(s) to the first question only
            if i == 0:
                if num_images == 1:
                    question = f"<image>\n{question}"
                elif num_images == 2:
                    question = f"<image> <image>\n{question}"
            
            conversations.append({
                "from": "human",
                "value": question
            })
            conversations.append({
                "from": "gpt",
                "value": qa.answer
            })
        
        return {
            "images": images,
            "conversations": conversations
        }
    
    @staticmethod
    def validate_conversation(conversation: Dict[str, Any]) -> bool:
        """Validate a conversation entry.
        
        Checks:
        - Has images and conversations fields
        - Conversations alternate between human/gpt
        - JSON answers parse correctly
        
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if "images" not in conversation or "conversations" not in conversation:
            return False
        
        if not conversation["images"] or not conversation["conversations"]:
            return False
        
        # Check conversation structure
        convs = conversation["conversations"]
        if len(convs) % 2 != 0:
            return False
        
        for i, turn in enumerate(convs):
            expected_from = "human" if i % 2 == 0 else "gpt"
            if turn.get("from") != expected_from:
                return False
            if "value" not in turn:
                return False
        
        # Validate JSON answers
        for i in range(1, len(convs), 2):
            answer = convs[i]["value"]
            # Check if it looks like JSON
            if answer.strip().startswith("{"):
                try:
                    parsed = json.loads(answer)
                    # Validate required fields based on type
                    if "type" not in parsed:
                        return False
                    
                    answer_type = parsed["type"]
                    if answer_type == "binary_detection":
                        if "artifact_present" not in parsed:
                            return False
                        if parsed["artifact_present"] not in ["yes", "no"]:
                            return False
                    elif answer_type == "localization":
                        if "coord_space" not in parsed or "bboxes" not in parsed:
                            return False
                        if parsed["coord_space"] != "pixel":
                            return False
                    elif answer_type == "global_explanation":
                        if "explanations" not in parsed:
                            return False
                        if not isinstance(parsed["explanations"], list):
                            return False
                except json.JSONDecodeError:
                    return False
        
        return True
    
    @staticmethod
    def save_to_json(
        conversations: List[Dict[str, Any]],
        output_path: str,
        validate: bool = True
    ):
        """Save conversations to JSON file.
        
        Args:
            conversations: List of conversation dictionaries
            output_path: Path to output JSON file
            validate: Whether to validate conversations before saving
        """
        if validate:
            valid_conversations = []
            for i, conv in enumerate(conversations):
                if VQASerializer.validate_conversation(conv):
                    valid_conversations.append(conv)
                else:
                    print(f"Warning: Conversation {i} failed validation and was skipped")
            conversations = valid_conversations
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(conversations)} conversations to {output_path}")

