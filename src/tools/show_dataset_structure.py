import os
import json
def list_files(startpath, max_images=3, snippet_length=50, image_extensions={'.jpg', '.jpeg', '.png'}):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")

        subindent = ' ' * 4 * (level + 1)
        image_count = 0

        for f in files:
            # Check if the file is an image
            if any(f.lower().endswith(ext) for ext in image_extensions):
                if image_count < max_images:
                    print(f"{subindent}{f}")
                    image_count += 1
                elif image_count == max_images:
                    print(f"{subindent}... more images")
                    image_count += 1
            else:
                # Print non-image files
                print(f"{subindent}{f}")
                if f.lower().endswith('.json'):
                    # Show structure of JSON file
                    try:
                        with open(os.path.join(root, f), 'r') as json_file:
                            data = json.load(json_file)
                            for key, value in data.items():
                                if isinstance(value, list) and len(value) > 0:
                                    print(f"{subindent}  {key}: list of length {len(value)}")
                                    first_item = value[0]
                                    if isinstance(first_item, dict):
                                        for subkey, subvalue in first_item.items():
                                            snippet = str(subvalue)[:snippet_length]
                                            if len(str(subvalue)) > snippet_length:
                                                snippet += ' ... '
                                                if type(subvalue) == type([]):
                                                    snippet += str(subvalue)[-snippet_length:]
                                            print(f"{subindent}    {subkey}: {type(subvalue).__name__} {snippet}")
                                elif isinstance(value, dict):
                                    print(f"{subindent}  {key}: dict with keys: {', '.join(value.keys())}")
                                else:
                                    print(f"{subindent}  {key}: {type(value).__name__}")
                    except Exception as e:
                        print(f"{subindent}  Error reading JSON file: {e}")

# Path to your dataset
dataset_root = r'../../datasets/StarSeg.v22i.coco-segmentation'

# Print the file tree showing non-image files and first 5 image files in each directory
list_files(dataset_root)