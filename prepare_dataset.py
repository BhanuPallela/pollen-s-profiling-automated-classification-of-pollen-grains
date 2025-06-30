import os, zipfile, shutil, random

# Setup paths
zip_path = 'pollen-grain-image-classification.zip'
extract_to = 'extracted_temp'
target_data_dir = 'data'
train_ratio = 0.8  # 80% train, 20% test

# Step 1: Extract ZIP
print("📦 Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print("✅ Extracted!")

# Step 2: Organize into train/test folders
print("📁 Organizing dataset...")

# Clear old data dir if exists
if os.path.exists(target_data_dir):
    shutil.rmtree(target_data_dir)

os.makedirs(os.path.join(target_data_dir, 'train'))
os.makedirs(os.path.join(target_data_dir, 'test'))

# Get all image files from extracted_temp
all_images = [f for f in os.listdir(extract_to) if f.lower().endswith('.jpg')]

# Group images by class (e.g. "anadenanthera", "arecaceae")
class_to_images = {}
for img in all_images:
    class_name = img.split('_')[0].split(' ')[0]  # Handles underscore or space
    class_to_images.setdefault(class_name, []).append(img)

# Split and move images
for class_name, images in class_to_images.items():
    random.shuffle(images)
    split_index = int(len(images) * train_ratio)
    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    # Create class folders
    train_class_dir = os.path.join(target_data_dir, 'train', class_name)
    test_class_dir = os.path.join(target_data_dir, 'test', class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_imgs:
        shutil.move(os.path.join(extract_to, img), os.path.join(train_class_dir, img))
    for img in test_imgs:
        shutil.move(os.path.join(extract_to, img), os.path.join(test_class_dir, img))

# Clean up
shutil.rmtree(extract_to)
print("✅ Data is ready in 'data/train/' and 'data/test/'")
