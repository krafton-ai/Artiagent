import os
import numpy as np
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Optional
import pathlib
import json
import glob
from PIL import Image


class COCODataLoader:
    """Handler for COCO dataset loading and image sampling"""
    
    def __init__(self, dataset_path: str, image_path: str):
        """
        Initialize COCO data loader
        
        Args:
            dataset_path: Path to COCO annotations directory
            image_path: Path to COCO images directory
        """
        self.dataset_path = dataset_path
        self.image_path = image_path
        
        # Load COCO annotations
        self.caption_file = os.path.join(dataset_path, "captions_train2017.json")
        self.class_file = os.path.join(dataset_path, "instances_train2017.json")
        
        self.coco_cap = COCO(self.caption_file)
        self.coco_class = COCO(self.class_file)
        
        # Get all image IDs
        self.image_ids = self.coco_cap.getImgIds()
    
    def get_category_ids(self, super_categories: List[str]) -> List[int]:
        """
        Get category IDs for given super categories
        
        Args:
            super_categories: List of super category names (e.g., ['person', 'animal'])
            
        Returns:
            List of category IDs
        """
        cat_ids = self.coco_class.getCatIds(supNms=super_categories)
        return cat_ids
    
    def get_category_names(self, cat_ids: List[int]) -> List[str]:
        """Get category names from category IDs"""
        cats = self.coco_class.loadCats(cat_ids)
        return [cat['name'] for cat in cats]
    
    def sample_image_by_category(self, cat_ids: List[int]) -> Tuple[Dict, np.ndarray, str]:
        """
        Sample a random image containing objects from specified categories
        
        Args:
            cat_ids: List of category IDs to sample from
            
        Returns:
            Tuple of (image_info, image_array, caption) with image dimensions adjusted to be divisible by 16
        """
        # Get images containing specified categories
        img_ids = self.coco_class.getImgIds(catIds=cat_ids[0])  # Use first category for sampling
        
        # Sample random image
        sampled_id = img_ids[np.random.randint(0, len(img_ids))]
        
        # Load image info and array
        img_info = self.coco_class.loadImgs(sampled_id)[0]
        
        # Load image with PIL
        img_path = os.path.join(self.image_path, img_info['file_name'])
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Rescale if shortest side is less than 480
        width, height = img.size
        if min(width, height) < 480:
            scale_factor = 480 / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Apply the same preprocessing as used for flux model
        # Ensure dimensions are divisible by 16 for correct input to the flux model
        shape = img_array.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        
        # Crop image to new dimensions
        img_array = img_array[:new_h, :new_w, :]
        
        # Get caption
        ann_ids = self.coco_cap.getAnnIds(imgIds=img_info['id'])
        anns = self.coco_cap.loadAnns(ann_ids)
        caption = anns[0]['caption'] if anns else ""
        
        return img_info, img_array, caption
    
    def get_image_categories(self, img_info: Dict) -> List[str]:
        """
        Get all category names present in an image
        
        Args:
            img_info: Image information dictionary
            
        Returns:
            List of unique category names in the image
        """
        # Get category information for the image
        ann_ids_class = self.coco_class.getAnnIds(imgIds=img_info['id'])
        anns_class = self.coco_class.loadAnns(ann_ids_class)
        
        # Extract category IDs from annotations
        cat_ids_in_image = [ann['category_id'] for ann in anns_class]
        
        # Get category names
        categories_in_image = []
        for cat_id in cat_ids_in_image:
            cat_info = self.coco_class.loadCats([cat_id])[0]
            categories_in_image.append(cat_info['name'])
        
        # Remove duplicates and return
        return list(set(categories_in_image))
    
    def load_image_by_info(self, img_info: Dict) -> np.ndarray:
        """
        Load image array from image info dictionary
        
        Args:
            img_info: COCO image info dictionary
            
        Returns:
            Image array with dimensions adjusted to be divisible by 16
        """
        img_path = os.path.join(self.image_path, img_info['file_name'])
        
        # Load image with PIL
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Rescale if shortest side is less than 480
        width, height = img.size
        if min(width, height) < 480:
            scale_factor = 480 / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Apply the same preprocessing as used for flux model
        # Ensure dimensions are divisible by 16 for correct input to the flux model
        shape = img_array.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        
        # Crop image to new dimensions
        img_array = img_array[:new_h, :new_w, :]
        
        return img_array
    
    def get_image_caption(self, img_info: Dict) -> str:
        """
        Get caption for a specific image
        
        Args:
            img_info: COCO image info dictionary
            
        Returns:
            Image caption string
        """
        ann_ids = self.coco_cap.getAnnIds(imgIds=img_info['id'])
        anns = self.coco_cap.loadAnns(ann_ids)
        caption = anns[0]['caption'] if anns else ""
        return caption
    
    def create_category_directories(self, category_names: List[str], base_path: str = 'data/coco_2017_extracted'):
        """Create directories for each category"""
        for category in category_names:
            pathlib.Path(f'{base_path}/{category}').mkdir(parents=True, exist_ok=True)


class ImageNetDataLoader:
    """Handler for ImageNet dataset loading and image sampling"""
    
    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        Initialize ImageNet data loader
        
        Args:
            dataset_path: Path to ImageNet dataset directory
            split: Dataset split ('train' or 'val')
        """
        self.dataset_path = dataset_path
        self.split = split
        self.split_path = os.path.join(dataset_path, split)
        
        # Load class mapping if available
        self.class_mapping = self._load_class_mapping()
        
        # Get all synset directories
        self.synsets = [d for d in os.listdir(self.split_path) 
                       if os.path.isdir(os.path.join(self.split_path, d))]
        
        # Build image index
        self._build_image_index()
    
    def _load_class_mapping(self) -> Dict[str, str]:
        """
        Load class mapping from synset IDs to human-readable names
        
        Returns:
            Dictionary mapping synset IDs to class names
        """
        mapping_files = [
            os.path.join(self.dataset_path, 'imagenet_class_index.json'),
            os.path.join(self.dataset_path, 'synset_words.txt'),
            os.path.join(self.dataset_path, 'LOC_synset_mapping.txt')
        ]
        
        class_mapping = {}
        
        # Try loading from JSON format first
        for mapping_file in mapping_files:
            if os.path.exists(mapping_file):
                if mapping_file.endswith('.json'):
                    with open(mapping_file, 'r') as f:
                        data = json.load(f)
                        for idx, (synset, name) in data.items():
                            class_mapping[synset] = name
                    break
                elif mapping_file.endswith('.txt'):
                    with open(mapping_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                synset = parts[0]
                                name = parts[1]
                                class_mapping[synset] = name
                    break
        
        return class_mapping
    
    def _build_image_index(self):
        """Build index of all images in the dataset"""
        self.image_index = {}
        
        for synset in self.synsets:
            synset_path = os.path.join(self.split_path, synset)
            image_files = []
            
            # Support common image formats
            for ext in ['*.JPEG', '*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(synset_path, ext)))
            
            self.image_index[synset] = image_files
    
    def get_class_names(self) -> List[str]:
        """
        Get all available class names
        
        Returns:
            List of class names (human-readable if mapping available, else synset IDs)
        """
        if self.class_mapping:
            return [self.class_mapping.get(synset, synset) for synset in self.synsets]
        else:
            return self.synsets
    
    def get_synsets(self) -> List[str]:
        """Get all available synset IDs"""
        return self.synsets
    
    def sample_image_by_class(self, class_names: List[str] = None, synsets: List[str] = None) -> Tuple[Dict, np.ndarray, str]:
        """
        Sample a random image from specified classes or synsets
        
        Args:
            class_names: List of human-readable class names to sample from
            synsets: List of synset IDs to sample from (takes precedence over class_names)
            
        Returns:
            Tuple of (image_info, image_array, class_name)
        """
        # Determine synsets to sample from
        if synsets:
            target_synsets = [s for s in synsets if s in self.synsets]
        elif class_names:
            # Convert class names to synsets
            target_synsets = []
            for class_name in class_names:
                for synset, mapped_name in self.class_mapping.items():
                    if mapped_name.lower() == class_name.lower() and synset in self.synsets:
                        target_synsets.append(synset)
        else:
            # Sample from all available synsets
            target_synsets = self.synsets
        
        if not target_synsets:
            raise ValueError("No matching synsets found for the specified classes")
        
        # Sample random synset
        sampled_synset = np.random.choice(target_synsets)
        
        # Sample random image from the synset
        if not self.image_index[sampled_synset]:
            raise ValueError(f"No images found for synset {sampled_synset}")
        
        sampled_image_path = np.random.choice(self.image_index[sampled_synset])
        
        # Load image
        img_array = self._load_and_preprocess_image(sampled_image_path)
        
        # Create image info
        img_info = {
            'file_name': os.path.basename(sampled_image_path),
            'file_path': sampled_image_path,
            'synset': sampled_synset,
            'class_name': self.class_mapping.get(sampled_synset, sampled_synset),
            'height': img_array.shape[0],
            'width': img_array.shape[1]
        }
        
        class_name = self.class_mapping.get(sampled_synset, sampled_synset)
        
        return img_info, img_array, class_name
    
    def load_image_by_path(self, image_path: str) -> np.ndarray:
        """
        Load image from file path with preprocessing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        return self._load_and_preprocess_image(image_path)
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for flux model compatibility
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array with dimensions adjusted to be divisible by 16
        """
        # Load image with PIL
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Rescale if shortest side is less than 480
        width, height = img.size
        if min(width, height) < 480:
            scale_factor = 480 / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Ensure dimensions are divisible by 16 for flux model compatibility
        shape = img_array.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        
        # Crop image to new dimensions
        img_array = img_array[:new_h, :new_w, :]
        
        return img_array
    
    def get_images_by_synset(self, synset: str) -> List[str]:
        """
        Get all image paths for a specific synset
        
        Args:
            synset: Synset ID
            
        Returns:
            List of image paths
        """
        return self.image_index.get(synset, [])
    
    def get_synset_stats(self) -> Dict[str, int]:
        """
        Get statistics about number of images per synset
        
        Returns:
            Dictionary mapping synset IDs to image counts
        """
        return {synset: len(images) for synset, images in self.image_index.items()}
    
    def create_class_directories(self, class_names: List[str], base_path: str = 'data/imagenet_extracted'):
        """
        Create directories for each class
        
        Args:
            class_names: List of class names or synsets
            base_path: Base directory to create class folders in
        """
        for class_name in class_names:
            # Use synset as folder name if it exists, otherwise use class name
            if class_name in self.synsets:
                folder_name = class_name
            else:
                # Find synset for class name
                folder_name = class_name
                for synset, mapped_name in self.class_mapping.items():
                    if mapped_name.lower() == class_name.lower():
                        folder_name = synset
                        break
            
            pathlib.Path(f'{base_path}/{folder_name}').mkdir(parents=True, exist_ok=True)


class CustomDirectoryDataLoader:
    """Handler for custom directory structure with images organized in subdirectories"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize custom directory data loader
        
        Args:
            dataset_path: Path to root directory containing subdirectories with images
                         Expected structure: dataset_path/class1/*.jpg, dataset_path/class2/*.jpg, etc.
        """
        self.dataset_path = dataset_path
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Build image index from subdirectories
        self._build_image_index()
        
        if not self.class_names:
            raise ValueError(f"No subdirectories with images found in {dataset_path}")
    
    def _build_image_index(self):
        """Build index of all images organized by class (subdirectory)"""
        self.image_index = {}
        self.class_names = []
        
        # Get all subdirectories
        subdirs = [d for d in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.dataset_path, subdir)
            image_files = []
            
            # Support common image formats
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', 
                       '*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF']:
                image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
            
            # Only include subdirectories that contain images
            if image_files:
                self.image_index[subdir] = image_files
                self.class_names.append(subdir)
        
        self.class_names.sort()  # Sort for consistent ordering
    
    def get_class_names(self) -> List[str]:
        """
        Get all available class names (subdirectory names)
        
        Returns:
            List of class names
        """
        return self.class_names.copy()
    
    def get_class_stats(self) -> Dict[str, int]:
        """
        Get statistics about number of images per class
        
        Returns:
            Dictionary mapping class names to image counts
        """
        return {class_name: len(images) for class_name, images in self.image_index.items()}
    
    def sample_image_by_class(self, class_names: List[str] = None) -> Tuple[Dict, np.ndarray, str]:
        """
        Sample a random image from specified classes
        
        Args:
            class_names: List of class names to sample from. If None, samples from all classes.
            
        Returns:
            Tuple of (image_info, image_array, class_name)
        """
        # Determine classes to sample from
        if class_names:
            target_classes = [c for c in class_names if c in self.class_names]
            if not target_classes:
                raise ValueError(f"No matching classes found. Available classes: {self.class_names}")
        else:
            target_classes = self.class_names
        
        # Sample random class
        sampled_class = np.random.choice(target_classes)
        
        # Sample random image from the class
        if not self.image_index[sampled_class]:
            raise ValueError(f"No images found for class {sampled_class}")
        
        sampled_image_path = np.random.choice(self.image_index[sampled_class])
        
        # Load and preprocess image
        img_array = self._load_and_preprocess_image(sampled_image_path)
        
        # Create image info
        img_info = {
            'file_name': os.path.basename(sampled_image_path),
            'file_path': sampled_image_path,
            'class_name': sampled_class,
            'height': img_array.shape[0],
            'width': img_array.shape[1]
        }
        
        return img_info, img_array, sampled_class
    
    def sample_images_from_class(self, class_name: str, num_samples: int = 1) -> List[Tuple[Dict, np.ndarray, str]]:
        """
        Sample multiple images from a specific class
        
        Args:
            class_name: Name of the class to sample from
            num_samples: Number of images to sample
            
        Returns:
            List of tuples (image_info, image_array, class_name)
        """
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {self.class_names}")
        
        available_images = self.image_index[class_name]
        if num_samples > len(available_images):
            raise ValueError(f"Requested {num_samples} samples but only {len(available_images)} images available for class '{class_name}'")
        
        # Sample without replacement
        sampled_paths = np.random.choice(available_images, size=num_samples, replace=False)
        
        results = []
        for image_path in sampled_paths:
            img_array = self._load_and_preprocess_image(image_path)
            img_info = {
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'class_name': class_name,
                'height': img_array.shape[0],
                'width': img_array.shape[1]
            }
            results.append((img_info, img_array, class_name))
        
        return results
    
    def get_all_images_from_class(self, class_name: str) -> List[str]:
        """
        Get all image paths for a specific class
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of image paths
        """
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {self.class_names}")
        
        return self.image_index[class_name].copy()
    
    def load_image_by_path(self, image_path: str) -> np.ndarray:
        """
        Load image from file path with preprocessing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        return self._load_and_preprocess_image(image_path)
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for flux model compatibility
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array with dimensions adjusted to be divisible by 16
        """
        # Load image with PIL
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Rescale if shortest side is less than 480
        width, height = img.size
        if min(width, height) < 480:
            scale_factor = 480 / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        
        # Ensure dimensions are divisible by 16 for flux model compatibility
        shape = img_array.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        
        # Crop image to new dimensions
        img_array = img_array[:new_h, :new_w, :]
        
        return img_array
    
    def create_class_directories(self, output_base_path: str):
        """
        Create output directories for each class
        
        Args:
            output_base_path: Base directory to create class folders in
        """
        for class_name in self.class_names:
            pathlib.Path(os.path.join(output_base_path, class_name)).mkdir(parents=True, exist_ok=True)
    
    def get_random_sample_from_each_class(self) -> List[Tuple[Dict, np.ndarray, str]]:
        """
        Get one random sample from each class
        
        Returns:
            List of tuples (image_info, image_array, class_name) for each class
        """
        results = []
        for class_name in self.class_names:
            img_info, img_array, class_name = self.sample_image_by_class([class_name])
            results.append((img_info, img_array, class_name))
        return results
    
    def get_image_caption(self, img_info: Dict) -> str:
        """
        Get caption for a specific image based on its class name
        
        Args:
            img_info: Image info dictionary containing class_name
            
        Returns:
            Simple caption string based on class name
        """
        class_name = img_info.get('class_name', 'unknown')
        return f"A photo of a {class_name}"