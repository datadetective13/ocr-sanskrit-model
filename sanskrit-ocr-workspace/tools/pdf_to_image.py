import os
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'image_{i + 1}.png')
        image.save(image_path, 'PNG')
        print(f'Saved: {image_path}')

if __name__ == "__main__":
    pdf_directory = '../data/raw/manuscripts'
    output_directory = '../data/processed/images'

    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_to_images(pdf_path, output_directory)