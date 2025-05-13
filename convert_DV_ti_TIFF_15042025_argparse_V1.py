import os
import bioformats
from tifffile import imwrite
import argparse  # Import the argparse module

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
    parser = argparse.ArgumentParser(description="Convert DeltaVision .dv files to TIFF format.")
    parser.add_argument("input_dir", help="Path to the input directory containing .dv files.")
    parser.add_argument("output_dir", help="Path to the output directory where TIFF files will be saved.")
    parser.add_argument("--multipage", action="store_true", help="Save each .dv file as a single multi-page TIFF.")
    parser.add_argument("--separate", action="store_false", dest="multipage", help="Save each frame as a separate TIFF file (default).")
    args = parser.parse_args()

    convert_dv_to_tiff(args.input_dir, args.output_dir, args.multipage)
    print("Conversion complete.")