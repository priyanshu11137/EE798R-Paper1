import numpy as np

# Function to load .npy file and count the number of annotations (people)
def count_people(npy_file_path):
    # Load the .npy file (it should contain an array of shape (N, 2) or (N, 1))
    annotations = np.load(npy_file_path)
    
    # Check the shape to ensure it's a valid annotation file
    if annotations.ndim == 2 and annotations.shape[1] == 2:
        # Count the number of dots (rows in the array)
        count = annotations.shape[0]
        return count
    else:
        raise ValueError("The provided file does not appear to contain valid dot annotations.")

if __name__ == "__main__":
    # Example: replace 'path_to_file.npy' with the actual file path
    npy_file_path = r'C:\Users\prj11\Downloads\EE798r\DM-Count\example_images\img_0220.npy'
    
    # Count people in the annotation file
    try:
        people_count = count_people(npy_file_path)
        print(f"Number of people (dot annotations): {people_count}")
    except Exception as e:
        print(f"Error: {e}")
