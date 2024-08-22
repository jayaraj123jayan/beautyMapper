import clip
import torch
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

# Load the model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess the image
# image = preprocess(Image.open("temp.jpeg"))

# Define the possible labels
labels = ["Matte", "Sheer", "Cream", "Liquid", "Satin"]

# # Encode the labels and the image
# text = clip.tokenize(labels)
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

# # Compute similarity
# logits_per_image, logits_per_text = model(image, text)
# probs = logits_per_image.softmax(dim=-1).cpu().numpy()

def resize_image_aspect_ratio(image, target_width):
    # Get the original dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = w / h
    
    # Calculate the new dimensions
    new_width = target_width
    new_height = int(new_width / aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_image

# Print the result
def predict_texture(img):
    # cv2.imwrite("lips_temp.jpeg", lip_image)
    # image = Image.open("lips_temp.jpeg")
    print("image received for texture prediction")
    image = Image.fromarray(resize_image_aspect_ratio(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),400))
    print("image converted")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    print("prepared input")
    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
    print("processed")
    # Get the similarity scores (logits)
    logits_per_image = outputs.logits_per_image  # shape: [batch_size, num_labels]

    # Convert logits to probabilities
    probs = logits_per_image.softmax(dim=1)  # shape: [batch_size, num_labels]

    max_prob_index = probs.argmax(dim=1).item()

    # Get the corresponding label
    most_probable_label = labels[max_prob_index]
    return most_probable_label
