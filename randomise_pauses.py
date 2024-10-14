import os
import random
import re

def process_text(text):
    # Find all occurrences of the placeholders (e.g., cat_1, cat_2, cat_5, etc.)
    cat_tags = re.findall(r'cat_\d+', text)
    
    # Remove all placeholders from the text
    text_without_cats = re.sub(r'cat_\d+', '', text)
    
    # Split the text into words (this will help to re-insert the tags at random places)
    words = text_without_cats.split()
    
    # Randomly insert the cat_tags back into the text
    for tag in cat_tags:
        insert_pos = random.randint(0, len(words))  # Choose a random position in the text
        words.insert(insert_pos, tag)
    
    # Join the words back into a single string
    modified_text = ' '.join(words)
    
    return modified_text

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            
            # Read the content of the text file
            with open(input_file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Process the text to remove and reinsert placeholders
            modified_text = process_text(text)
            
            # Write the modified text to a new file in the output folder
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(modified_text)
    
    print(f"Processed files have been saved to {output_folder}")

# Example usage:
input_folder = "/Users/madalina/Documents/M1TAL/stage_GC/fichiersavecpausescat"  # Replace with the path to your folder containing the text files
output_folder = "/Users/madalina/Documents/M1TAL/stage_GC/randomised_fichiersavecpausescat"  # Replace with the path to the output folder

process_folder(input_folder, output_folder)
