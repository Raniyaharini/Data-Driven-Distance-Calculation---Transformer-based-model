import os
import cv2

def repair_corrupted_jpeg(input_folder, output_folder):
    try:
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate through all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Attempt to repair the current JPEG file
                image = cv2.imread(input_path)
                if image is not None:
                    cv2.imwrite(output_path, image)
                    print(f"Repaired: {filename}")
                else:
                    print(f"Failed to repair: {filename}")

        print("Repair process completed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_folder = r"E:\\Thesis\\all_images"  # Replace with the path to your folder of corrupted JPEGs
    output_folder = r"E:\\Thesis\\all"  # Replace with the desired output folder path

    repair_corrupted_jpeg(input_folder, output_folder)