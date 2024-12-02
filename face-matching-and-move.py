import os
import argparse
import face_recognition
from PIL import Image
import numpy as np
import dlib

def encode_face_cnn(image_path, downscale_factor=1):
    """
    Encodes all faces in the given image file.

    Args:
        image_path (str): Path to the image file.
        downscale_factor (int): Factor by which the image should be downscaled.

    Returns:
        list: List of face encodings found in the image.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        small_image = Image.fromarray(image).resize(
            (image.shape[1] // downscale_factor, image.shape[0] // downscale_factor),
            Image.Resampling.LANCZOS
        )
        small_image_array = np.array(small_image)
        face_locations = face_recognition.face_locations(small_image_array, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(small_image_array, face_locations, model="large")
        return face_encodings
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []
    
def encode_face(image_path, downscale_factor=1):
    """
    Encodes all faces in the given image file.

    Args:
        image_path (str): Path to the image file.
        downscale_factor (int): Factor by which the image should be downscaled.

    Returns:
        list: List of face encodings found in the image.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0)
        face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
        return face_encodings
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []    
    
def encode_face_2(image_path):
    """
    Detects and encodes all faces in the given image file using dlib.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: List of face encodings found in the image.
    """
    try:
        # Load the image
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_np = np.array(image)

        # Initialize dlib face detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

        # Detect faces
        faces = detector(image_np, 1)  # 1 means upsample once for better detection

        # Encode faces
        face_encodings = []
        for face in faces:
            shape = predictor(image_np, face)
            face_encoding = np.array(face_rec_model.compute_face_descriptor(image_np, shape))
            face_encodings.append(face_encoding)

        return face_encodings

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []    

def compare_faces_with_known(known_encoding, unknown_image_path, tolerance, downscale_factor):
    """
    Compares a known face encoding with all face encodings in an unknown image.

    Args:
        known_encoding (list): Encoding of the known face.
        unknown_image_path (str): Path to the unknown image file.
        tolerance (float): Threshold for face comparison.
        downscale_factor (int): Factor by which the image should be downscaled.

    Returns:
        bool: True if any face in the image matches the known encoding, otherwise False.
    """
    # unknown_encodings = encode_face(unknown_image_path, downscale_factor)
    unknown_encodings = encode_face(unknown_image_path)
    for idx, unknown_encoding in enumerate(unknown_encodings):
        match_result = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)[0]
        print(f"Face index {idx}: Match result = {match_result}")
        if match_result:
            return True
    return False

def load_processed_files(log_file):
    """
    Loads the list of already processed files from the log file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        set: A set of processed file paths.
    """
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def log_processed_file(log_file, file_path):
    """
    Logs a processed file to the log file.

    Args:
        log_file (str): Path to the log file.
        file_path (str): Path of the processed file.
    """
    with open(log_file, 'a') as f:
        f.write(file_path + '\n')

def scan_and_move(known_encoding, source_folder, destination_folder, tolerance, downscale_factor, valid_extensions, log_file):
    """
    Scans all images in a folder (including subfolders), compares faces,
    and moves matching images to a destination folder using os.rename.

    Args:
        known_encoding (list): Encoding of the known face.
        source_folder (str): Root folder to scan for images.
        destination_folder (str): Folder to move matching images to.
        tolerance (float): Threshold for face comparison.
        downscale_factor (int): Factor by which the image should be downscaled.
        valid_extensions (tuple): Supported image file extensions.
        log_file (str): Path to the log file.
    """
    processed_files = load_processed_files(log_file)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    count =0
    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(valid_extensions) and file_path not in processed_files:
                print(f"Processing: {file_path}")
                log_processed_file(log_file, file_path)
                # count +=1
                # if count>1000:
                #     return
                if compare_faces_with_known(known_encoding, file_path, tolerance, downscale_factor):
                    # Log the file as processed
                    destination_path = os.path.join(destination_folder, os.path.basename(file_path))
                    try:
                        os.rename(file_path, destination_path)
                        print(f"Moved to: {destination_path}")
                    except Exception as e:
                        print(f"Error moving file {file_path} to {destination_path}: {e}")

                

def main():
    """
    Main function to parse arguments, encode the known face, scan for matches, and move matching images.
    """
    parser = argparse.ArgumentParser(description="Face recognition and matching script with resumable processing.")
    parser.add_argument("--downscale_factor", type=int, default=2, help="Downscale factor for image processing.")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images to scan.")
    parser.add_argument("--destination_folder", type=str, required=True, help="Folder to store matched images.")
    parser.add_argument("--tolerance", type=float, default=0.4, help="Tolerance for face matching.")
    parser.add_argument("--valid_extensions", type=str, nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"], help="Valid image file extensions.")
    parser.add_argument("--known_image_path", type=str, required=True, help="Path to the known face image.")
    parser.add_argument("--log_file", type=str, default="processed_files.log", help="Log file to track processed files.")

    args = parser.parse_args()

    # Encode the known face
    # known_encoding = encode_face(args.known_image_path, args.downscale_factor)
    known_encoding = encode_face(args.known_image_path)
    if not known_encoding:
        print("No face encodings found in the known image. Exiting.")
        return
    known_encoding = known_encoding[0]

    # Scan and move matching images
    scan_and_move(
        known_encoding, 
        args.image_folder, 
        args.destination_folder, 
        args.tolerance, 
        args.downscale_factor, 
        tuple(args.valid_extensions),
        args.log_file
    )

if __name__ == "__main__":
    main()

