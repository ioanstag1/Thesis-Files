
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import random
import os
from hovertrans import create_model
from config import config

def load_model(fold):
    args = config()
    args.mode = 'eval'
    model = create_model(
        img_size=args.img_size,
        num_classes=args.class_num,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_inner_head=args.num_inner_head
    )
    weight_path = f"/content/drive/MyDrive/RUNS/weight_2024-07-23_13-38-16/hovertrans/GDPHSYSUCC/{fold}/bestmodel.pth"
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model, args.img_size

def generate_lime(model, img_path, img_size, num_samples=1000):
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img_np = np.array(img)

    def predict(images):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        transform = transforms.Normalize(mean=[0.2706, 0.2671, 0.2874], std=[0.1808, 0.1793, 0.1904])
        images = transform(images)
        with torch.no_grad():
            outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_np, predict, top_labels=2, hide_color=0, num_samples=num_samples)
    probabilities = predict(np.expand_dims(img_np, axis=0))
    predicted_label = np.argmax(probabilities)

    return img_np, explanation, predicted_label, probabilities

def generate_superpixels(img_np, n_segments=50, compactness=10, sigma=1):
    segments = slic(img_np, n_segments=n_segments, compactness=compactness, sigma=sigma)
    return segments

def plot_comparison(original_image, explanation, segments, label, image_name, fold, predicted_label, original_label, probabilities, output_folder):
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    ax[0].imshow(original_image)
    ax[0].set_title(f"Original Image\nTrue Label: {original_label}")
    ax[0].axis('off')
    
    ax[1].imshow(mark_boundaries(original_image, segments))
    ax[1].set_title("Superpixels with Boundaries")
    ax[1].axis('off')
    
    ax[2].imshow(segments, cmap='nipy_spectral')
    ax[2].set_title("Colored Superpixels")
    ax[2].axis('off')
    
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=True)
    ax[3].imshow(mark_boundaries(temp / 255.0, mask))
    ax[3].set_title("LIME Explanation (Top 10)")
    ax[3].axis('off')
    
    temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)
    ax[4].imshow(mark_boundaries(temp / 255.0, mask))
    ax[4].set_title("Positive and Negative Features (Top 10)")
    ax[4].axis('off')
    
    plt.suptitle(f"Fold {fold}, Predicted Label: {predicted_label}, Probabilities: {probabilities}")
    
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, f"{image_name}_fold{fold}_label{predicted_label}.png")
    plt.savefig(output_path)
    plt.show()  # Display the plot
    plt.close(fig)

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    images_folder = "/content/drive/MyDrive/data/img"
    output_folder = "/content/drive/MyDrive/RUNS/weight_2024-07-23_13-38-16/output"  # Output folder to save plots
    os.makedirs(output_folder, exist_ok=True)
    
    all_image_names = os.listdir(images_folder)
    #target_image_names = random.sample(all_image_names, 50)  # Pick 50 random images
    target_image_names = [
     "malignant(280).png", "malignant(786).png"
    ]

    for image_name in target_image_names:
        img_path = os.path.join(images_folder, image_name)
        if os.path.exists(img_path):
            original_label = "malignant" if "malignant" in image_name else "benign"
            for fold in range(5):
                model, img_size = load_model(fold)

                original_image_np, explanation, predicted_label, probabilities = generate_lime(model, img_path, img_size)
                segments = generate_superpixels(original_image_np, n_segments=100, compactness=10, sigma=1)
                plot_comparison(original_image_np, explanation, segments, explanation.top_labels[0], image_name, fold, predicted_label, original_label, probabilities, output_folder)

                print(f"Processed {img_path} for fold {fold}, Predicted Label: {predicted_label}, Probabilities: {probabilities}")
        else:
            print(f"Image {img_path} not found.")

