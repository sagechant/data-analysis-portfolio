import os
import bioformats
from tifffile import imwrite

def convert_dv_to_tiff(input_dir, output_dir, save_as_multipage=True):
    """
    Converts DeltaVision OMX TIRF raw files (.dv) in the input directory
    to TIFF format in the output directory.

    Args:
        input_dir (str): Path to the directory containing the .dv files.
        output_dir (str): Path to the directory where TIFF files will be saved.
        save_as_multipage (bool, optional): If True, saves all frames from a .dv file
                                            into a single multi-page TIFF file.
                                            Defaults to False (saving each frame as a separate TIFF).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".dv"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            try:
                print(f"Processing file: {input_path}")
                with bioformats.ImageReader(input_path) as reader:
                    num_frames = reader.image_count
                    if save_as_multipage:
                        all_frames = [reader.read(image=i) for i in range(num_frames)]
                        output_filename = f"{base_name}.tif"
                        output_path = os.path.join(output_dir, output_filename)
                        imwrite(output_path, all_frames)
                        print(f"Saved {num_frames} frames to multi-page TIFF: {output_path}")
                    else:
                        for i in range(num_frames):
                            image_data = reader.read(image=i)
                            output_filename = f"{base_name}_frame_{i+1:03d}.tif"
                            output_path = os.path.join(output_dir, output_filename)
                            imwrite(output_path, image_data)
                            print(f"Saved frame {i+1} to: {output_path}")

                    # You can access metadata using bioformats if needed:
                    # metadata = bioformats.get_omexml_metadata(input_path)
                    # print(metadata)

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "/path/to/your/raw/data"  # Replace with the actual input directory
    output_directory = "/path/to/your/tiff/data" # Replace with the desired output directory
    save_multipage_tiff = True  # Set to True to save as a single multi-page TIFF per .dv file

    convert_dv_to_tiff(input_directory, output_directory, save_multipage_tiff)
    print("Conversion complete.")