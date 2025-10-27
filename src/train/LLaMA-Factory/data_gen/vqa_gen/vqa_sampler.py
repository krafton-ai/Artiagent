import random
from typing import List, Tuple
from .types import ArtiInstance
from .vqa_builders import VQABuilders, QAPair

class VQASampler:
    """Sample and assemble Q-A pairs with dropout and shuffling."""
    
    def __init__(
        self,
        format_dropout: float = 0.15,
        qa_dropout_rate: float = 0.3,
        single_turn_prob: float = 0.2,
        always_start_with_binary: bool = False
    ):
        """Initialize VQA sampler.
        
        Args:
            format_dropout: Probability to omit format instructions (0.1-0.2)
            qa_dropout_rate: Probability to drop each Q-A pair (0.2-0.4)
            single_turn_prob: Probability to create single-turn conversation (0.2)
            always_start_with_binary: If True, always start artifact conversations with 1.1 (binary detection)
        """
        self.format_dropout = format_dropout
        self.qa_dropout_rate = qa_dropout_rate
        self.single_turn_prob = single_turn_prob
        self.always_start_with_binary = always_start_with_binary
    
    def sample_real_image(self, instance: ArtiInstance) -> Tuple[List[str], List[QAPair]]:
        """Sample Q-A pairs for real image only.
        
        Returns:
            Tuple of (image_list, qa_pairs)
        """
        images = [instance.real_image]
        qa_pairs = []
        
        # Only binary detection is applicable for real images
        qa_pairs.append(VQABuilders.build_binary_detection_real(self.format_dropout))
        
        return images, qa_pairs
    
    def sample_artifact_image(self, instance: ArtiInstance) -> Tuple[List[str], List[QAPair]]:
        """Sample Q-A pairs with fixed contextual order.
        
        Order: 1.1 → 1.2 → 1.4 → 1.3
        Random start from {1.1, 1.2, 1.3}, then continue forward (no wrap)
        
        Returns:
            Tuple of (image_list, qa_pairs)
        """
        images = [instance.artifact_image]
        
        # Choose random start point (or force 1.1 if configured)
        if self.always_start_with_binary:
            start = '1.1'
        else:
            start_options = ['1.1', '1.2', '1.3']
            start = random.choice(start_options)
        
        qa_pairs = []
        has_bboxes = bool(instance.artifacts)
        
        if start == '1.1':
            # [1.1, 1.2, 1.4×K, 1.3]
            qa_pairs.append(VQABuilders.build_binary_detection_artifact(self.format_dropout))
            
            if has_bboxes:
                qa_pairs.append(VQABuilders.build_localization(instance.artifacts, self.format_dropout))
                # 1.4 Regional (one per artifact)
                for artifact in instance.artifacts:
                    qa_pairs.append(VQABuilders.build_regional_explanation(artifact))
            
            qa_pairs.append(VQABuilders.build_global_explanation(
                metadata_caption=instance.metadata_caption,
                artifacts=instance.artifacts,
                format_dropout=self.format_dropout
            ))
        
        elif start == '1.2':
            # [1.2, 1.4×K, 1.3]
            if has_bboxes:
                qa_pairs.append(VQABuilders.build_localization(instance.artifacts, self.format_dropout))
                # 1.4 Regional
                for artifact in instance.artifacts:
                    qa_pairs.append(VQABuilders.build_regional_explanation(artifact))
            
            qa_pairs.append(VQABuilders.build_global_explanation(
                metadata_caption=instance.metadata_caption,
                artifacts=instance.artifacts,
                format_dropout=self.format_dropout
            ))
        
        else:  # start == '1.3'
            # [1.3]
            qa_pairs.append(VQABuilders.build_global_explanation(
                metadata_caption=instance.metadata_caption,
                artifacts=instance.artifacts,
                format_dropout=self.format_dropout
            ))
        
        return images, qa_pairs
    
    def sample_pair(self, instance: ArtiInstance) -> Tuple[List[str], List[QAPair]]:
        """Sample ONE pairwise task (4.1-4.4) as single-turn.
        
        Tasks: 4.1 Binary, 4.2 Localization, 4.3 Regional, 4.4 Explanation
        
        Returns:
            Tuple of (image_list, qa_pairs with exactly one element)
        """
        # Randomly decide image order
        if random.random() < 0.5:
            images = [instance.artifact_image, instance.real_image]
            artifact_position = "first"
        else:
            images = [instance.real_image, instance.artifact_image]
            artifact_position = "second"
        
        # Collect available tasks
        available_tasks = []
        
        # 4.1 Binary - always available
        available_tasks.append(('4.1', VQABuilders.build_pair_binary(artifact_position)))
        
        # 4.2 Localization - requires bboxes
        if instance.artifacts:
            task = VQABuilders.build_pair_localization(instance.artifacts, artifact_position)
            if task:
                available_tasks.append(('4.2', task))
        
        # 4.3 Regional - requires bboxes
        if instance.artifacts:
            task = VQABuilders.build_pair_regional(instance.artifacts, artifact_position)
            if task:
                available_tasks.append(('4.3', task))
        
        # 4.4 Explanation - requires caption
        if instance.metadata_caption:
            task = VQABuilders.build_pair_explanation(instance.metadata_caption, artifact_position)
            if task:
                available_tasks.append(('4.4', task))
        
        # Randomly select ONE task
        if available_tasks:
            task_name, qa_pair = random.choice(available_tasks)
            return images, [qa_pair]
        else:
            # Fallback to 4.1 if nothing else available
            return images, [VQABuilders.build_pair_binary(artifact_position)]
    
    def sample_conversation(self, instance: ArtiInstance, mode: str = "auto") -> Tuple[List[str], List[QAPair]]:
        """Sample a complete conversation for an ArtiInstance.
        
        Args:
            instance: ArtiInstance to generate conversation for
            mode: Sampling mode - "real", "artifact", "pair", or "auto"
        
        Returns:
            Tuple of (image_list, qa_pairs)
        """
        if mode == "auto":
            # Automatically determine mode based on available images
            has_real = instance.real_image is not None
            has_artifact = instance.artifact_image is not None and len(instance.artifacts) > 0
            
            if has_real and has_artifact:
                # Randomly choose between single image or pair
                mode = random.choice(["real", "artifact", "pair"])
            elif has_artifact:
                mode = "artifact"
            elif has_real:
                mode = "real"
            else:
                return [], []
        
        # Sample based on mode
        if mode == "real":
            images, qa_pairs = self.sample_real_image(instance)
        elif mode == "artifact":
            images, qa_pairs = self.sample_artifact_image(instance)
        elif mode == "pair":
            images, qa_pairs = self.sample_pair(instance)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # No shuffling or dropout - maintain fixed order
        return images, qa_pairs

