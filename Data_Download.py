# 1. Install necessary libraries
!pip install torch torchvision torchaudio transformers open_clip_torch datasets accelerate pycocoevalcap nltk scst

# 2. Define the target directory
DATA_DIR = "./data/coco"
!mkdir -p $DATA_DIR
!mkdir -p $DATA_DIR/annotations # Create subdirectory for annotations

# 3. Direct Download using official COCO links (wget)
print("\nStarting COCO download from official source...")

# Download Validation Images (Smaller file: ~1GB)
!wget -c http://images.cocodataset.org/zips/val2017.zip -P $DATA_DIR

# Download Annotations (Caption and other annotations: ~241MB)
!wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P $DATA_DIR

# Download Training Images (Large file: ~18GB)
!wget -c http://images.cocodataset.org/zips/train2017.zip -P $DATA_DIR

# 4. Unzip the downloaded files
print("\nStarting COCO extraction...")

# Unzip annotation files into the 'annotations' subdirectory
!unzip $DATA_DIR/annotations_trainval2017.zip -d $DATA_DIR/annotations

# Unzip image files
!unzip $DATA_DIR/val2017.zip -d $DATA_DIR
!unzip $DATA_DIR/train2017.zip -d $DATA_DIR

# 5. Clean up zip files to save disk space in Colab
!rm $DATA_DIR/*.zip

print("\nCOCO dataset successfully downloaded and extracted to:", DATA_DIR)
# Verify the structure:
!ls $DATA_DIR/annotations/annotations