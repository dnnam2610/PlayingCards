import os
import glob
import random
import cv2
import shutil
import time
from tqdm.auto import tqdm

if __name__ == "__main__":
    print("Strating augmentation process !!!")
    start_time = time.time()

    input_train_images = "datasets/PlayingCards/train/images"
    input_train_labels = "datasets/PlayingCards/train/labels"
    input_val_images = "datasets/PlayingCards/valid/images"
    input_val_labels = "datasets/PlayingCards/valid/labels"
    input_test_images = "datasets/PlayingCards/test/images"
    input_test_labels = "datasets/PlayingCards/test/labels"

    output_train_images = "datasets_augmented/PlayingCards/train/images"
    output_train_labels = "datasets_augmented/PlayingCards/train/labels"
    output_val_images = "datasets_augmented/PlayingCards/valid/images"
    output_val_labels = "datasets_augmented/PlayingCards/valid/labels"
    output_test_images = "datasets_augmented/PlayingCards/test/images"
    output_test_labels = "datasets_augmented/PlayingCards/test/labels"

    os.makedirs(output_train_images, exist_ok=True)
    os.makedirs(output_train_labels, exist_ok=True)
    os.makedirs(output_val_images, exist_ok=True)
    os.makedirs(output_val_labels, exist_ok=True)
    os.makedirs(output_test_images, exist_ok=True)
    os.makedirs(output_test_labels, exist_ok=True)


    image_files = glob.glob(os.path.join(input_train_images, "*.jpg"))
    image_files = [os.path.basename(f) for f in image_files]

    # Take out 50% of images
    num_images = len(image_files)
    num_to_augment = num_images // 2
    augment_images = random.sample(image_files, num_to_augment)

    # Copy training data to new data
    progress_bar = tqdm(image_files)
    progress_bar.set_description(f"Copying training set: ")
    for img_file in progress_bar:
        shutil.copy(os.path.join(input_train_images, img_file), os.path.join(output_train_images, img_file))
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(input_train_labels, label_file)):
            shutil.copy(os.path.join(input_train_labels, label_file), os.path.join(output_train_labels, label_file))

    # Copy val data to new data
    val_image_files = glob.glob(os.path.join(input_val_images, "*.jpg"))
    progress_bar = tqdm(val_image_files)
    progress_bar.set_description(f"Copying valid set: ")
    for img_path in progress_bar:
        img_file = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_val_images, img_file))
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(input_val_labels, label_file)):
            shutil.copy(os.path.join(input_val_labels, label_file), os.path.join(output_val_labels, label_file))

    # Copy test data to new data
    test_image_files = glob.glob(os.path.join(input_test_images, "*.jpg"))
    progress_bar = tqdm(test_image_files)
    progress_bar.set_description(f"Copying testing set: ")
    for img_path in progress_bar:
        img_file = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(output_test_images, img_file))
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(input_test_labels, label_file)):
            shutil.copy(os.path.join(input_test_labels, label_file), os.path.join(output_val_labels, label_file))

    # Gaussian Blur
    progress_bar = tqdm(augment_images)
    progress_bar.set_description(f"Augmentation: ")
    for img_file in progress_bar:
        img_path = os.path.join(input_train_images, img_file)
        img = cv2.imread(img_path)

        blurred_img = cv2.GaussianBlur(img,(7,7),2)

        blurred_img_path = os.path.join(output_train_images, f"blurred_{img_file}")
        cv2.imwrite(blurred_img_path, blurred_img)

        # Copy label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(input_train_labels, label_file)):
            shutil.copy(
                os.path.join(input_train_labels, label_file),
                os.path.join(output_train_labels, f"blurred_{label_file}")
            )
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    print("FINISH !!!")