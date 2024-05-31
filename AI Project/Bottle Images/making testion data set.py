import os
import shutil
from random import sample

# Define paths
original_data_dir = 'Bottle Images'
train_dir = 'Bottle Images'
test_dir = 'Testing'
classes = ['Plastic', 'Soda', 'Water']
num_test_images = 500

# Create directories if they don't exist
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Move images
for cls in classes:
    class_dir = os.path.join(original_data_dir, cls)
    train_class_dir = os.path.join(train_dir, cls)
    test_class_dir = os.path.join(test_dir, cls)
    
    # Get all files in the class directory
    all_images = os.listdir(class_dir)
    
    # Sample 500 images for the test set
    test_images = sample(all_images, num_test_images)
    
    # Move test images
    for img in test_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(test_class_dir, img))
    
    # Move remaining images to the train directory
    remaining_images = os.listdir(class_dir)
    for img in remaining_images:
        shutil.move(os.path.join(class_dir, img), os.path.join(train_class_dir, img))

print("Images have been successfully moved to train and test directories.")
