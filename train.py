import chess
import chess.svg
import cairosvg
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import random
import shutil
import logging
from pathlib import Path
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm


# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)


# Set up logging with detailed messages
def setup_logging(log_dir):
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(filename=log_file)],
    )
    logging.info("Logging setup complete. Logs will be written to {}".format(log_file))


# Create directories and log their creation
def setup_directories(data_dir):
    # Check if directories already exist and contain data
    for dir_name in ["train", "test", "val"]:
        dir_path = data_dir.joinpath(dir_name)
        if dir_path.exists() and any(dir_path.iterdir()):
            logging.warning(f"Directory {dir_path} already exists and contains data. Please clean up or use a new directory.")
            return None
        dir_path.mkdir(parents=True, exist_ok=True)

    logging.info("Directories 'train', 'test' and 'val' created in {}".format(data_dir))
    return data_dir


# Augment images and log the process
def augment_images(img, num_augmented_images):
    aug = iaa.Sequential(
        [
            iaa.Affine(rotate=(-25, 25)),
            iaa.Flipud(0.5),
            iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Crop(percent=(0, 0.2)),
            iaa.LinearContrast((0.75, 1.5)),
        ]
    )
    logging.info("Image augmentation setup complete")
    return np.array([aug.augment_image(img) for _ in range(num_augmented_images)])


# Generate positions, save images, and log the process
def generate_position(i, data_dir):
    try:
        board = chess.Board()
        for _ in range(np.random.randint(1, 50)):
            if not board.is_game_over():
                move = random.choice(list(board.legal_moves))
                board.push(move)

        # Validate FEN string
        if not chess.Board(board.fen()).is_valid():
            logging.warning(f"Invalid FEN string generated for position {i}. Skipping.")
            return None

        fen = board.fen()
        safe_fen = fen.replace("/", "_")

        output_file = data_dir / f"chess_position_{i}_{safe_fen}.jpg"  # Save as JPEG
        if output_file.exists():
            logging.warning(
                f"Image {output_file} already exists. Skipping position {i}."
            )
            return None

        svg_data = chess.svg.board(board=board)

        with open(str(output_file), "wb") as f:
            cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=f)

        # Convert PNG to JPEG
        png = Image.open(str(output_file))
        png.load() # required for png.split()

        if png.mode == 'RGBA':
            background = Image.new("RGB", png.size, (255, 255, 255))
            background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            background.save(str(output_file), 'JPEG', quality=80)
        else:
            png.convert('RGB').save(str(output_file), 'JPEG', quality=80)

        logging.info(f"Generated image for position {i} and saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error occurred while generating position {i}: {type(e).__name__}, {e}")
        # Ensure output_file is defined before attempting to check if it exists
        if 'output_file' in locals() and output_file.exists():
            os.remove(output_file)
            logging.info(f"Removed failed file {output_file}")
        return None

def generate_position_with_data_dir(i):
    return generate_position(i, data_dir)

# Use multiprocessing to generate images and log the process
def generate_images(num_positions, data_dir):
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        with tqdm(total=num_positions) as pbar:
            for i, _ in enumerate(pool.imap_unordered(generate_position_with_data_dir, range(num_positions))):
                pbar.update()
        files = list(tqdm(pool.imap(generate_position_with_data_dir, range(num_positions)), total=num_positions))
    logging.info(f"Generated {num_positions} images")
    return [f for f in files if f is not None]

# Split file paths and log the process
def split_files(files, train_test_ratio, val_ratio):  # Added validation ratio
    train_files, temp_files = train_test_split(
        files, test_size=1 - train_test_ratio, random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=val_ratio/(1-train_test_ratio), random_state=42
    )  # Split remaining files into validation and test sets
    logging.info(
        f"Split files into {len(train_files)} training, {len(val_files)} validation and {len(test_files)} test files"
    )
    return train_files, val_files, test_files  # Return validation files


# Augment positions, save images, and log the process
def augment_and_save_images(files, image_size, num_augmented_images, data_dir, folder):
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        for file_path in tqdm(files, desc="Augmenting images"):
            try:
                with Image.open(file_path) as img:
                    img = np.array(img.resize(image_size))
                    aug_imgs = pool.map(augment_images, [img]*num_augmented_images)
                    aug_img_paths = [
                        data_dir.joinpath(folder, f"{file_path.stem}_{idx}.jpg")  # Save as JPEG in the correct subdirectory
                        for idx in range(num_augmented_images)
                    ]
                    for aug_img, aug_img_path in zip(aug_imgs, aug_img_paths):
                        with Image.fromarray(aug_img) as img:
                            img.save(aug_img_path, "JPEG", quality=85)  # Save as JPEG
                logging.info(
                    f"Augmented and saved {num_augmented_images} images for {file_path}"
                )
            except Exception as e:
                logging.error(
                    f"Error occurred while augmenting and saving image {file_path}: {type(e).__name__}, {e}"
                )
                for aug_img_path in aug_img_paths:
                    if aug_img_path.exists():
                        os.remove(aug_img_path)
                        logging.info(f"Removed failed file {aug_img_path}")


# Move the files to the respective directories and log the process
def move_files(files, data_dir, folder):
    for file_path in tqdm(files, desc=f"Moving files to {folder}"):
        try:
            file_path.rename(data_dir.joinpath(folder, file_path.name))
            logging.info(f"Moved file {file_path} to {folder}")
        except Exception as e:
            logging.error(f"Error occurred while moving {folder} file {file_path}: {type(e).__name__}, {e}")


if __name__ == "__main__":
    log_dir = Path("log")
    setup_logging(log_dir)
    data_dir = setup_directories(Path("data"))
    if data_dir is None:  # Check if setup_directories returned None
        logging.error("Data directory setup failed. Exiting.")
        exit(1)
    num_positions = 5000
    files = generate_images(num_positions, data_dir)
    train_test_ratio = 0.7  # Adjusted train-test ratio
    val_ratio = 0.15  # Added validation ratio
    train_files, val_files, test_files = split_files(files, train_test_ratio, val_ratio)  # Split into train, validation, and test sets
    logging.info(f"{len(train_files)} training samples, {len(val_files)} validation samples, {len(test_files)} test samples")  # Log validation samples
    image_size = (224, 224)
    num_augmented_images = 5
    augment_and_save_images(train_files, image_size, num_augmented_images, data_dir, "train")
    augment_and_save_images(val_files, image_size, num_augmented_images, data_dir, "val")  # Augment validation images
    augment_and_save_images(test_files, image_size, num_augmented_images, data_dir, "test")
    move_files(train_files, data_dir, "train")
    move_files(val_files, data_dir, "val")  # Move validation files
    move_files(test_files, data_dir, "test")
    logging.info("Training, validation, and test files are moved to the respective directories")  # Log validation directory
