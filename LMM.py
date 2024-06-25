import os
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

def load_model_and_processor(model_name):
    """Load the processor and model given the model name."""
    processor = Pix2StructProcessor.from_pretrained(model_name)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    return processor, model

def generate_response(processor, model, image, question):
    """Generate a response from the model given an image and a question."""
    inputs = processor(images=image, text=question, return_tensors="pt")
    predictions = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(predictions[0], skip_special_tokens=True)
    return response

def process_images_and_questions(image_dir, questions, output_file):
    """Process images and questions, then write responses to the output file."""
    image_extensions = ['.png', '.jpg', '.jpeg']

    # Load models and processors
    processor_matcha, model_matcha = load_model_and_processor('google/matcha-chartqa')
    processor_pix2struct, model_pix2struct = load_model_and_processor('google/pix2struct-chartqa-base')

    with open(output_file, 'w') as out_file:
        # Iterate over all files in the directory
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_dir, filename)
                
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")
                    continue

                for question in questions:
                    # Generate responses from both models
                    try:
                        response_matcha = generate_response(processor_matcha, model_matcha, image, question)
                    except Exception as e:
                        response_matcha = f"Error generating response with Matcha model: {e}"

                    try:
                        response_pix2struct = generate_response(processor_pix2struct, model_pix2struct, image, question)
                    except Exception as e:
                        response_pix2struct = f"Error generating response with Pix2Struct model: {e}"

                    # Format and write the output
                    output = (
                        f"Image: {filename}\n"
                        f"Question: {question}\n"
                        f"Response Matcha: {response_matcha}\n"
                        f"Response Pix2Struct: {response_pix2struct}\n\n"
                    )

                    print(output)
                    out_file.write(output)

if __name__ == "__main__":
    # Load questions
    with open('questions.txt', 'r') as file:
        questions = [line.strip() for line in file]

    # Path to the directory containing images
    image_dir = "images"
    
    # Path to the output file
    output_file = 'responses3.txt'
    
    # Process images and questions
    process_images_and_questions(image_dir, questions, output_file)


