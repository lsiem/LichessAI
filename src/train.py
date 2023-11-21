"""
This module contains functions for generating, augmenting, and splitting chess position images for training, validation, and testing.
"""

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
from multiprocessing import Pool, Value
import multiprocessing
from tqdm import tqdm
import io


# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

data_dir = Path("data").resolve()

def setup_logging(log_dir):
    """
    Sets up logging for the application. This function configures the logging module to output log messages to a file and the console.
    It creates a log file in the specified directory and sets the log level to INFO.
    
    Args:
        log_dir (str): The directory where the log file will be stored. This directory will be created if it does not exist.

    Returns:
        None
    """
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file),
            console_handler
        ],
    )
    logging.info("Logging setup complete. Logs will be written to {}".format(log_file))



# Sets up the train, test, and validation directories within the given data directory. Checks if directories already exist and contain data, warns if they do, creates them if needed. Logs status messages. Returns the data directory.
def setup_directories(data_dir):
    """
    Sets up the train, test, and validation directories within the given data directory.
    This function checks if the directories already exist and contain data. If they do, a warning is logged and None is returned.
    If the directories do not exist, they are created.
    
    Args:
        data_dir (str): The directory where the data will be stored. This directory will be created if it does not exist.

    Returns:
        str: The absolute path to the data directory, or None if the directories already exist and contain data.
    """
    data_dir = Path(data_dir).resolve()
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
class ImageGenerator:
    def augment_images(self, img, num_augmented_images):
        """
        Augments images and logs the process.
        This function applies a series of transformations to the input image, including rotation, noise addition, cropping, contrast adjustment, and brightness adjustment.
        The transformations are applied in parallel using multiprocessing to speed up the process.
        
        Args:
            img (numpy.ndarray): The image to be augmented. This should be a 3D array representing an RGB image.
            num_augmented_images (int): The number of augmented images to generate. This should be a positive integer.

        Returns:
            numpy.ndarray: A 4D array containing the augmented images. The first dimension is the number of images, and the other three dimensions are the image dimensions.
        """
        aug = iaa.Sequential(
            [
                iaa.Affine(rotate=(-25, 25)),
                # Removed flipping as it can alter the orientation of chess pieces
                iaa.AdditiveGaussianNoise(scale=(10, 60)),
                iaa.Crop(percent=(0, 0.2)),
                iaa.LinearContrast((0.75, 1.5)),
                iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
                iaa.GammaContrast((0.8, 1.2)),  # Contrast adjustment
                iaa.CropAndPad(percent=(-0.25, 0.25)),  # Random cropping
            ]
        )

        # Use multiprocessing to parallelize image augmentation
        with Pool(multiprocessing.cpu_count()) as p:
            augmented_images = list(
                tqdm(
                    p.imap(aug.augment_image, [img] * num_augmented_images),
                    total=num_augmented_images,
                    desc="Augmenting images",
                )
            )

        logging.info(f"Augmented {num_augmented_images} images.")
        return np.array(augmented_images)



    # Generates a random chess position as a JPEG image file.
    def generate_position(self, i, data_dir):
        """
        Generates a random chess position as a JPEG image file.
        This function creates a random chess position by making a series of random legal moves on a chess board.
        It then converts the board to a FEN string and saves it as a .fen file.
        The board is also converted to an SVG image, which is then converted to a JPEG image and saved as a .jpg file.
        
        Args:
            i (int): The index of the position. This is used to name the output files.
            data_dir (str): The directory where the image and FEN file will be stored. This directory will be created if it does not exist.

        Returns:
            str: The path to the output JPEG file, or None if an error occurred.
        """
        data_dir = Path(data_dir).resolve()
        output_file = None
        logging.info(f"Starting to generate position {i}")
        board = chess.Board()
        for _ in range(np.random.randint(1, 50)):
            if not board.is_game_over():
                move = random.choice(list(board.legal_moves))
                board.push(move)

        try:
            fen = board.fen()
        except chess.BoardError as e:
            logging.error(f"Error occurred while generating FEN for position {i}: {type(e).__name__}, {e}")
            return None

        # Validate FEN string
        if not chess.Board(fen).is_valid():
            logging.warning(f"Invalid FEN string generated for position {i}. Skipping.")
            return None

        fen_file = data_dir / f"chess_position_{i}_{safe_fen}.fen"  # Save as FEN
        output_file = data_dir / f"chess_position_{i}_{safe_fen}.jpg"  # Save as JPEG
        fen_file.write_text(fen)
        if output_file.exists():
            logging.warning(
                f"Image {output_file} already exists. Skipping position {i}."
            )
            return None

        try:
            # Convert board to SVG
            svg_data = chess.svg.board(board=board)
        except chess.svg.SvgError as e:
            logging.error(f"Error occurred while converting board to SVG for position {i}: {type(e).__name__}, {e}")
            return None

        try:
            # Directly convert SVG to JPEG
            jpeg_data = cairosvg.svg2jpeg(bytestring=svg_data.encode("utf-8"))
            logging.info(f"Converted SVG to JPEG for position {i}")
        except cairosvg.Error as e:
            logging.error(f"Error occurred while converting SVG to JPEG for position {i}: {type(e).__name__}, {e}")
            return None

        try:
            # Save JPEG to disk
            with open(str(output_file), "wb") as f:
                f.write(jpeg_data)
            logging.info(f"Generated image for position {i} and saved to {output_file}")
        except OSError as e:
            logging.error(f"Error occurred while saving JPEG for position {i}: {type(e).__name__}, {e}")
            if output_file.exists():
                os.remove(output_file)
                logging.info(f"Removed failed file {output_file}")
            output_file = None
        return output_file

    def generate_position_with_data_dir(self, i):
        """
        Wrapper function for generate_position that uses the global data_dir variable.
        
        Args:
            i (int): The index of the position.
            
        Returns:
            str: The path to the output file.
        """
        return self.generate_position(i, data_dir)


    def generate_images(self, num_positions, data_dir, save_examples=False):
        """
        Generates a specified number of chess position images.
        This function generates a specified number of random chess positions and saves them as JPEG images.
        It uses multiprocessing to speed up the process.
        If requested, it also saves a few examples of the generated images for inspection.
        
        Args:
            num_positions (int): The number of positions to generate. This should be a positive integer.
            data_dir (str): The directory where the images will be stored. This directory will be created if it does not exist.
            save_examples (bool, optional): Whether to save examples of the generated images. If True, the first 10 images are copied to an 'examples/generated' directory. Defaults to False.

        Returns:
            list: The paths to the generated images. If an error occurred while generating an image, its path is not included in the list.
        """
        data_dir = Path(data_dir).resolve()
        num_processes = min(multiprocessing.cpu_count(), num_positions)
        counter = Value('i', 0)
        with tqdm(total=num_positions, file=open(os.devnull, 'w')) as pbar:
            with Pool(processes=num_processes) as pool:
                for i, output_file in enumerate(pool.imap_unordered(self.generate_position_with_data_dir, range(num_positions))):
                    if output_file is not None:
                        output_path = Path(output_file)
                        if output_file.is_file():
                            with output_path.open("rb") as f:
                                with counter.get_lock():
                                    counter.value += 1
                                    pbar.update(counter.value)
                                logging.info(f"Successfully generated image for position {i}")
                        else:
                            logging.warning(f"Failed to generate image for position {i}")
                files = [f for f in pool.imap(self.generate_position_with_data_dir, range(num_positions)) if f is not None and Path(f).is_file()]
        logging.info(f"Generated {len(files)} images out of {num_positions} requested")
        
        # Save some examples for inspection
        if save_examples:
            example_dir = Path("examples/generated").resolve()
            example_dir.mkdir(parents=True, exist_ok=True)
            for i, file in enumerate(files[:10]):
                shutil.copy(file, example_dir / f"example_{i}.jpg")
            logging.info(f"Saved examples of generated images to {example_dir}")
        
        return files

# Split file paths and log the process
def split_files(files, train_test_ratio, val_ratio):  # Added validation ratio
    """
    Splits the files into training, validation, and test sets.
    This function splits a list of files into three sets: training, validation, and test.
    The sizes of the sets are determined by the specified ratios.
    The splitting is done randomly but in a deterministic way, so the same split can be reproduced if the function is called with the same arguments.
    
    Args:
        files (list): The files to be split. This should be a list of file paths.
        train_test_ratio (float): The ratio of training files to total files. This should be a number between 0 and 1.
        val_ratio (float): The ratio of validation files to total files. This should be a number between 0 and 1.

    Returns:
        tuple: Three lists containing the training, validation, and test files, respectively.
    """
    train_files, temp_files = train_test_split(
        files, test_size=1 - train_test_ratio, random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=val_ratio/(1-train_test_ratio), random_state=42
    )  # Split remaining files into validation and test sets
    logging.info(
    aug = iaa.Sequential(
        [
            iaa.Affine(rotate=(-25, 25)),
            # Removed flipping as it can alter the orientation of chess pieces
            iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Crop(percent=(0, 0.2)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
            iaa.GammaContrast((0.8, 1.2)),  # Contrast adjustment
            iaa.CropAndPad(percent=(-0.25, 0.25)),  # Random cropping
        ]
    )

    # Use multiprocessing to parallelize image augmentation
    with Pool(multiprocessing.cpu_count()) as p:
        augmented_images = list(
            tqdm(
                p.imap(aug.augment_image, [img] * num_augmented_images),
                total=num_augmented_images,
                desc="Augmenting images",
            )
        )

    logging.info(f"Augmented {num_augmented_images} images.")
    return np.array(augmented_images)



# Generates a random chess position as a JPEG image file.
        def generate_position(self, i, data_dir):
    """
    Generates a random chess position as a JPEG image file.
    This function creates a random chess position by making a series of random legal moves on a chess board.
    It then converts the board to a FEN string and saves it as a .fen file.
    The board is also converted to an SVG image, which is then converted to a JPEG image and saved as a .jpg file.
    
    Args:
        i (int): The index of the position. This is used to name the output files.
        data_dir (str): The directory where the image and FEN file will be stored. This directory will be created if it does not exist.

    Returns:
        str: The path to the output JPEG file, or None if an error occurred.
    """
    data_dir = Path(data_dir).resolve()
    output_file = None
    logging.info(f"Starting to generate position {i}")
    board = chess.Board()
    for _ in range(np.random.randint(1, 50)):
        if not board.is_game_over():
            move = random.choice(list(board.legal_moves))
            board.push(move)

    try:
        fen = board.fen()
    except chess.BoardError as e:
        logging.error(f"Error occurred while generating FEN for position {i}: {type(e).__name__}, {e}")
        return None

    # Validate FEN string
    if not chess.Board(fen).is_valid():
        logging.warning(f"Invalid FEN string generated for position {i}. Skipping.")
        return None

    fen_file = data_dir / f"chess_position_{i}_{safe_fen}.fen"  # Save as FEN
    output_file = data_dir / f"chess_position_{i}_{safe_fen}.jpg"  # Save as JPEG
    fen_file.write_text(fen)
    if output_file.exists():
        logging.warning(
            f"Image {output_file} already exists. Skipping position {i}."
        )
        return None

    try:
        # Convert board to SVG
        svg_data = chess.svg.board(board=board)
    except chess.svg.SvgError as e:
        logging.error(f"Error occurred while converting board to SVG for position {i}: {type(e).__name__}, {e}")
        return None

    try:
        # Directly convert SVG to JPEG
        jpeg_data = cairosvg.svg2jpeg(bytestring=svg_data.encode("utf-8"))
        logging.info(f"Converted SVG to JPEG for position {i}")
    except cairosvg.Error as e:
        logging.error(f"Error occurred while converting SVG to JPEG for position {i}: {type(e).__name__}, {e}")
        return None

    try:
        # Save JPEG to disk
        with open(str(output_file), "wb") as f:
            f.write(jpeg_data)
        logging.info(f"Generated image for position {i} and saved to {output_file}")
    except OSError as e:
        logging.error(f"Error occurred while saving JPEG for position {i}: {type(e).__name__}, {e}")
        if output_file.exists():
            os.remove(output_file)
            logging.info(f"Removed failed file {output_file}")
        output_file = None
    return output_file

        def generate_position_with_data_dir(self, i):
    """
    Wrapper function for generate_position that uses the global data_dir variable.
    
    Args:
        i (int): The index of the position.
        
    Returns:
        str: The path to the output file.
    """
    return generate_position(i, data_dir)


        def generate_images(self, num_positions, data_dir, save_examples=False):
    """
    Generates a specified number of chess position images.
    This function generates a specified number of random chess positions and saves them as JPEG images.
    It uses multiprocessing to speed up the process.
    If requested, it also saves a few examples of the generated images for inspection.
    
    Args:
        num_positions (int): The number of positions to generate. This should be a positive integer.
        data_dir (str): The directory where the images will be stored. This directory will be created if it does not exist.
        save_examples (bool, optional): Whether to save examples of the generated images. If True, the first 10 images are copied to an 'examples/generated' directory. Defaults to False.

    Returns:
        list: The paths to the generated images. If an error occurred while generating an image, its path is not included in the list.
    """
    data_dir = Path(data_dir).resolve()
    num_processes = min(multiprocessing.cpu_count(), num_positions)
    counter = Value('i', 0)
    with tqdm(total=num_positions, file=open(os.devnull, 'w')) as pbar:
        with Pool(processes=num_processes) as pool:
            for i, output_file in enumerate(pool.imap_unordered(generate_position_with_data_dir, range(num_positions))):
                if output_file is not None:
                    output_path = Path(output_file)
                    if output_file.is_file():
                        with output_path.open("rb") as f:
                            with counter.get_lock():
                                counter.value += 1
                                pbar.update(counter.value)
                            logging.info(f"Successfully generated image for position {i}")
                    else:
                            logging.warning(f"Failed to generate image for position {i}")
            files = [f for f in pool.imap(generate_position_with_data_dir, range(num_positions)) if f is not None and Path(f).is_file()]
    logging.info(f"Generated {len(files)} images out of {num_positions} requested")
    
    # Save some examples for inspection
    if save_examples:
        example_dir = Path("examples/generated").resolve()
        example_dir.mkdir(parents=True, exist_ok=True)
        for i, file in enumerate(files[:10]):
            shutil.copy(file, example_dir / f"example_{i}.jpg")
        logging.info(f"Saved examples of generated images to {example_dir}")
    
    return files

# Split file paths and log the process
def split_files(files, train_test_ratio, val_ratio):  # Added validation ratio
    """
    Splits the files into training, validation, and test sets.
    This function splits a list of files into three sets: training, validation, and test.
    The sizes of the sets are determined by the specified ratios.
    The splitting is done randomly but in a deterministic way, so the same split can be reproduced if the function is called with the same arguments.
    
    Args:
        files (list): The files to be split. This should be a list of file paths.
        train_test_ratio (float): The ratio of training files to total files. This should be a number between 0 and 1.
        val_ratio (float): The ratio of validation files to total files. This should be a number between 0 and 1.

    Returns:
        tuple: Three lists containing the training, validation, and test files, respectively.
    """
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


def augment_and_save_images(files, image_size, num_augmented_images, data_dir, folder, save_examples=False):
    """
    Augments and saves images.
    This function opens each file in the input list, resizes the image to the specified size, and generates a specified number of augmented versions of the image.
    The augmented images are saved in the specified directory and subdirectory.
    If requested, it also saves a few examples of the augmented images for inspection.
    
    Args:
        files (list): The files to be augmented. This should be a list of file paths.
        image_size (tuple): The size of the images. This should be a tuple of two integers representing the width and height of the images.
        num_augmented_images (int): The number of augmented images to generate. This should be a positive integer.
        data_dir (str): The directory where the images will be stored. This directory will be created if it does not exist.
        folder (str): The subdirectory where the images will be stored. This directory will be created if it does not exist.
        save_examples (bool, optional): Whether to save examples of the augmented images. If True, the first 10 images are copied to an 'examples/augmented' directory. Defaults to False.

    Returns:
        None
    """
    data_dir = Path(data_dir).resolve()
    total_augmentations = len(files) * num_augmented_images
    with tqdm(total=total_augmentations, desc="Augmenting images") as pbar:
        for file_path in files:
            aug_img_paths = []
            if not os.path.exists(file_path):
                logging.error(f"File {file_path} does not exist. Skipping.")
                continue
            try:
                with Image.open(file_path) as img:
                    img = np.array(img.resize(image_size))
            except OSError as e:
                logging.error(f"Error occurred while opening image file {file_path}: {type(e).__name__}, {e}")
                return

            try:
                img = np.reshape(img, (image_size[0], image_size[1], 3))  # Reshape the image
                img = img.astype('uint8')  # Ensure the image is in the correct data type
            except ValueError as e:
                logging.error(f"Error occurred while reshaping or converting data type of image {file_path}: {type(e).__name__}, {e}")
                return

            try:
                aug_imgs = augment_images(img, num_augmented_images)
            except Exception as e:  # Catch all exceptions as the augment_images function can raise various types of exceptions
                logging.error(f"Error occurred while augmenting image {file_path}: {type(e).__name__}, {e}")
                return

            aug_img_paths = [
                data_dir.joinpath(folder, f"{file_path.stem}_{idx}.jpg")
                for idx in range(num_augmented_images)
            ]
            for aug_img, aug_img_path in zip(aug_imgs, aug_img_paths):
                try:
                    with Image.fromarray(aug_img) as img:
                        img.save(aug_img_path, "JPEG", quality=85)
                except OSError as e:
                    logging.error(f"Error occurred while saving augmented image {aug_img_path}: {type(e).__name__}, {e}")
                    if aug_img_path.exists():
                        os.remove(aug_img_path)
                        logging.info(f"Removed failed file {aug_img_path}")
                    continue

            logging.info(
                f"Augmented and saved {num_augmented_images} images for {file_path}"
            )
            pbar.update(num_augmented_images)  # Update the progress bar here

            # Save some examples for inspection
            if save_examples and file_path.stem.endswith("_0"):
                example_dir = Path("examples/augmented").resolve()
                example_dir.mkdir(parents=True, exist_ok=True)
                for i, aug_img_path in enumerate(aug_img_paths[:10]):
                    shutil.copy(aug_img_path, example_dir / f"example_{i}.jpg")
                logging.info(f"Saved examples of augmented images to {example_dir}")



# Move the files to the respective directories and log the process
def move_files(files, data_dir, folder):
    """
    Moves the files to the respective directories and logs the process.
    This function moves each file in the input list to the specified directory and subdirectory.
    It logs a message for each file that is moved, and returns a list of the new paths of the moved files.
    
    Args:
        files (list): The files to be moved. This should be a list of file paths.
        data_dir (str): The directory where the files will be moved. This directory will be created if it does not exist.
        folder (str): The subdirectory where the files will be moved. This directory will be created if it does not exist.

    Returns:
        list: The new paths of the moved files.
    """
    data_dir = Path(data_dir).resolve()
    new_paths = []
    for file_path in tqdm(files, desc=f"Moving files to {folder}", file=open(os.devnull, 'w')):
        try:
            new_path = data_dir.joinpath(folder, file_path.name)
            shutil.move(str(file_path), str(new_path))
            logging.info(f"Moved file {file_path} to {folder}")
            new_paths.append(new_path)
        except Exception as e:
            logging.error(f"Error occurred while moving {folder} file {file_path}: {type(e).__name__}, {e}")
    return new_paths


def main():
    """
    The main function of the script. Sets up logging and directories, generates and augments images, splits them into training, validation, and test sets, and moves them to their respective directories.
    This function is the entry point of the script. It performs the following steps:
    1. Sets up logging.
    2. Sets up the data directory and its subdirectories for training, validation, and test data.
    3. Generates a specified number of random chess position images.
    4. Splits the generated images into training, validation, and test sets according to the specified ratios.
    5. Moves the images to their respective directories.
    6. Augments the images in each set and saves the augmented images in the same directories.
    7. Deletes all files in the data directory after moving them into their respective subdirectories.
    """
    log_dir = Path("log").resolve()
    setup_logging(log_dir)
    data_dir = setup_directories(Path("data").resolve())
    if data_dir is None:  # Check if setup_directories returned None
        logging.error("Data directory setup failed. Exiting.")
        exit(1)
    num_positions = 50
    image_generator = ImageGenerator()
    files = image_generator.generate_images(num_positions, data_dir)

    # Verify that the files were generated
    if not files:
        logging.error("No files generated. Exiting.")
        exit(1)
    train_test_ratio = 0.7  # Adjusted train-test ratio
    val_ratio = 0.15  # Added validation ratio
    train_files, val_files, test_files = split_files(files, train_test_ratio, val_ratio)  # Split into train, validation, and test sets
    logging.info(f"{len(train_files)} training samples, {len(val_files)} validation samples, {len(test_files)} test samples")  # Log validation samples
    image_size = (224, 224)
    num_augmented_images = 5
    # Move files into their respective directories and update paths
    train_files = move_files(train_files, data_dir, "train")
    val_files = move_files(val_files, data_dir, "val")
    test_files = move_files(test_files, data_dir, "test")
    # Augment images in their respective directories
    augment_and_save_images(train_files, image_size, num_augmented_images, data_dir, "train", save_examples=True)
    augment_and_save_images(val_files, image_size, num_augmented_images, data_dir, "val", save_examples=True)  # Augment validation images
    augment_and_save_images(test_files, image_size, num_augmented_images, data_dir, "test", save_examples=True)
    logging.info("Training, validation, and test files are moved to the respective directories")  # Log validation directory
    # Delete all files in the data directory after moving them into their respective sub directories
    for file in data_dir.glob('*'):
        if file.is_file():
            file.unlink()
    logging.info("All files in the data directory have been deleted after moving them into their respective sub directories")

if __name__ == "__main__":
    main()

