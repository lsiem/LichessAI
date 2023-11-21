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
    # function body

def setup_directories(data_dir):
    # function body

def augment_images(img, num_augmented_images):
    # function body

def generate_position(i, data_dir):
    # function body

def generate_images(num_positions, data_dir):
    # function body

def split_files(files, train_test_ratio, val_ratio):
    # function body

def augment_and_save_images(files, image_size, num_augmented_images, data_dir, folder):
    # function body

def move_files(files, data_dir, folder):
    # function body