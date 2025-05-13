import os
from ij import IJ, ImagePlus
from ij.io import DirectoryChooser

# Get the input directory
input_dc = DirectoryChooser("Choose the input directory containing .dv files")
input_dir = input_dc.getDirectory()
if not input_dir:
    print("Input directory not selected. Script will exit.")
    exit()
print("Selected input directory: {}".format(input_dir))

# Get the output directory
output_dc = DirectoryChooser("Choose the output directory for TIFF files")
output_dir = output_dc.getDirectory()
if not output_dir:
    print("Output directory not selected. Script will exit.")
    exit()
print("Selected output directory: {}".format(output_dir))

if not os.path.exists(input_dir):
    print("Error: Input directory does not exist.")
else:
    found_files = False
    # Process each .dv file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".dv"):
            found_files = True
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = "{}.tif".format(base_name)
            output_path = os.path.join(output_dir, output_filename)

            print("Processing file: {}".format(input_path))
            try:
                # Open the file
                print("Trying to open: {}".format(input_path))
                imp = IJ.openImage(input_path)
                if imp:
                    # Save as TIFF
                    print("Trying to save: {} as {}".format(input_path, output_path))
                    IJ.saveAs(imp, "Tiff", output_path)
                    # imp.close() # Commenting out close due to errors
                    print("Saved as: {}".format(output_path))
                else:
                    print("Error: Could not open file: {}".format(input_path))
            except Exception as e:
                print("Error processing {}: {}".format(input_path, e))

    if not found_files:
        print("No .dv files found in the input directory.")

print("Conversion complete.")