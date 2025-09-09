import os
import numpy as np
from pycocotools.coco import COCO
from typing import List, Dict, Tuple, Optional, Union
import pathlib
import json
import glob
from PIL import Image
import logging
from typing import Any


def preprocess_image_for_flux(image_path_or_pil: Union[str, Image.Image]) -> np.ndarray:
    """
    Shared image preprocessing function for flux model compatibility
    
    Args:
        image_path_or_pil: Either a file path to image or PIL Image object
        
    Returns:
        Image array with dimensions adjusted to be divisible by 16
    """
    # Load image with PIL if path provided
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil
        
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
        
        # Load and preprocess image
        img_path = os.path.join(self.image_path, img_info['file_name'])
        img_array = preprocess_image_for_flux(img_path)
        
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
        return preprocess_image_for_flux(img_path)
    
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
        return preprocess_image_for_flux(image_path)
    
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
    """Handler for custom directory structure with images directly in a single directory"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize custom directory data loader
        
        Args:
            dataset_path: Path to directory containing images directly
                         Expected structure: dataset_path/*.jpg, dataset_path/*.png, etc.
        """
        self.dataset_path = dataset_path
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Build image index from directory
        self._build_image_index()
        
        if not self.image_paths:
            raise ValueError(f"No images found in {dataset_path}")
    
    def _build_image_index(self):
        """Build index of all images in the directory"""
        self.image_paths = []
        
        # Support common image formats
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', 
                   '*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF']:
            self.image_paths.extend(glob.glob(os.path.join(self.dataset_path, ext)))
        
        self.image_paths.sort()  # Sort for consistent ordering
    
    def get_image_count(self) -> int:
        """
        Get total number of images in the directory
        
        Returns:
            Number of images
        """
        return len(self.image_paths)
    
    def get_all_image_paths(self) -> List[str]:
        """
        Get all image paths in the directory
        
        Returns:
            List of image paths
        """
        return self.image_paths.copy()
    
    def sample_random_image(self) -> Tuple[Dict, np.ndarray]:
        """
        Sample a random image from the directory
        
        Returns:
            Tuple of (image_info, image_array)
        """
        if not self.image_paths:
            raise ValueError("No images available to sample")
        
        # Sample random image path
        sampled_image_path = np.random.choice(self.image_paths)
        
        # Load and preprocess image
        img_array = self._load_and_preprocess_image(sampled_image_path)
        
        # Create image info
        img_info = {
            'file_name': os.path.basename(sampled_image_path),
            'file_path': sampled_image_path,
            'height': img_array.shape[0],
            'width': img_array.shape[1]
        }
        
        return img_info, img_array
    
    def sample_multiple_images(self, num_samples: int = 1) -> List[Tuple[Dict, np.ndarray]]:
        """
        Sample multiple images from the directory
        
        Args:
            num_samples: Number of images to sample
            
        Returns:
            List of tuples (image_info, image_array)
        """
        if num_samples > len(self.image_paths):
            raise ValueError(f"Requested {num_samples} samples but only {len(self.image_paths)} images available")
        
        # Sample without replacement
        sampled_paths = np.random.choice(self.image_paths, size=num_samples, replace=False)
        
        results = []
        for image_path in sampled_paths:
            img_array = self._load_and_preprocess_image(image_path)
            img_info = {
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'height': img_array.shape[0],
                'width': img_array.shape[1]
            }
            results.append((img_info, img_array))
        
        return results
    
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
        return preprocess_image_for_flux(image_path)
    
    def load_image_by_info(self, img_info: Dict) -> np.ndarray:
        """
        Load image by image info dictionary
        
        Args:
            img_info: Dictionary containing 'file_path' key
            
        Returns:
            Preprocessed image array
        """
        image_path = img_info.get('file_path')
        if not image_path:
            raise ValueError("Image info must contain 'file_path' key")
        return self._load_and_preprocess_image(image_path)
    

def _get_coco_image_list(
    data_loader: COCODataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None,
    max_instances_per_image: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get image list for COCO dataset with optional filtering.
    
    Args:
        data_loader: COCO data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        max_instances_per_image: Maximum instances per image for filtering
        
    Returns:
        List of image information dictionaries
    """
    cat_ids = data_loader.get_category_ids(categories)
    image_list = []
    image_ids_seen = set()
    
    # Count instances per image if filtering is requested
    instance_counts = {}
    if max_instances_per_image is not None:
        print("Counting instances per image...")
        from collections import defaultdict
        instance_counts = defaultdict(int)
        for ann in data_loader.coco_class.dataset['annotations']:
            image_id = ann['image_id']
            instance_counts[image_id] += 1
    
    for cat_id in cat_ids:
        img_ids = data_loader.coco_class.getImgIds(catIds=[cat_id])
        for img_id in img_ids:
            if img_id not in image_ids_seen:
                # Filter by instance count if specified
                if max_instances_per_image is not None:
                    if instance_counts[img_id] >= max_instances_per_image:
                        continue
                
                img_info = data_loader.coco_class.loadImgs([img_id])[0]
                image_list.append(img_info)
                image_ids_seen.add(img_id)
                
    print("number of images", len(image_list))
    return image_list


def _get_imagenet_image_list(
    data_loader: ImageNetDataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get image list for ImageNet dataset.
    
    Args:
        data_loader: ImageNet data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        
    Returns:
        List of image information dictionaries
    """
    # Determine target synsets
    target_synsets = []
    for class_name in categories:
        for synset, mapped_name in data_loader.class_mapping.items():
            if mapped_name.lower() == class_name.lower() and synset in data_loader.synsets:
                target_synsets.append(synset)
    
    if not target_synsets:
        target_synsets = data_loader.synsets
    
    image_list = []
    for synset in target_synsets:
        image_paths = data_loader.get_images_by_synset(synset)
        for img_path in image_paths:
            img_info = {
                'id': hash(img_path) % 1000000,  # Generate unique ID
                'file_name': os.path.basename(img_path),
                'file_path': img_path,
                'synset': synset,
                'class_name': data_loader.class_mapping.get(synset, synset)
            }
            image_list.append(img_info)
            
            if max_images and len(image_list) >= max_images:  
                break
        if max_images and len(image_list) >= max_images:
            break
    
    return image_list


def _get_custom_image_list(
    data_loader: CustomDirectoryDataLoader, 
    categories: List[str], 
    max_images: Optional[int] = None,
    logger: logging.Logger = None
) -> List[Dict[str, Any]]:
    """
    Get image list for custom dataset.
    
    Args:
        data_loader: Custom directory data loader instance
        categories: List of categories (ignored for flat directory structure)
        max_images: Maximum number of images to process
        logger: Logger instance
        
    Returns:
        List of image information dictionaries
    """
    # Get all available image paths from the directory
    all_image_paths = data_loader.get_all_image_paths()
    if logger:
        logger.info(f"Found {len(all_image_paths)} images in custom dataset directory")
    
    # Limit images if max_images is specified
    if max_images and max_images < len(all_image_paths):
        all_image_paths = all_image_paths[:max_images]
        if logger:
            logger.info(f"Limited to first {max_images} images")
    
    # Create image info list
    image_list = []
    for img_path in all_image_paths:
        img_info = {
            'id': hash(img_path) % 1000000,  # Generate unique ID
            'file_name': os.path.basename(img_path),
            'file_path': img_path
        }
        image_list.append(img_info)
    
    return image_list


def _get_image_list(
    dataset_type: str, 
    data_loader: Any, 
    categories: List[str], 
    max_images: Optional[int] = None,
    max_instances_per_image: Optional[int] = None,
    logger: logging.Logger = None
) -> List[Dict[str, Any]]:
    """
    Get image list based on dataset type.
    
    Args:
        dataset_type: Type of dataset
        data_loader: Data loader instance
        categories: List of categories to process
        max_images: Maximum number of images to process
        max_instances_per_image: Maximum number of instances per image
        logger: Logger instance
        
    Returns:
        List of image information dictionaries
    """
    if dataset_type == "coco":
        return _get_coco_image_list(data_loader, categories, max_images, max_instances_per_image)
    elif dataset_type == "imagenet":
        return _get_imagenet_image_list(data_loader, categories, max_images)
    elif dataset_type == "custom":
        return _get_custom_image_list(data_loader, categories, max_images, logger)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _initialize_data_loader(dataset_type: str, config: Dict[str, Any]) -> Any:
    """
    Initialize the appropriate data loader based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('coco', 'imagenet', 'custom')
        config: Configuration dictionary
        
    Returns:
        Initialized data loader instance
    """
    if dataset_type == "coco":
        return COCODataLoader(config['dataset_path'], config['image_path'])
    elif dataset_type == "imagenet":
        return ImageNetDataLoader(config['dataset_path'], config['imagenet_split'])
    elif dataset_type == "custom":
        return CustomDirectoryDataLoader(config['dataset_path'])
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")