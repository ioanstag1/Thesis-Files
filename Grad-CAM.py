# Grad-CAM++ for a model with single output, all stages
import torch
from torchvision import transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
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
    weight_path = f"/content/drive/MyDrive/weight_2024-08-20_23-49-44/hovertrans/GDPHSYSUCC/{fold}/bestmodel.pth"   
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model, args.img_size

def generate_gradcam(model, img_path, target_layers, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(cv2.resize(img, (img_size, img_size))) / 255
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2706, 0.2671, 0.2874], std=[0.1808, 0.1793, 0.1904])
    ])(img).unsqueeze(0)

    cams = []
    for target_layer in target_layers:
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        cams.append(cam_image)

    # Get the model prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()

    return img, cams, predicted_label

def plot_comparison(original_image, cam_images, image_id, fold, predicted_label, stage):
    label_text = 'malignant' if predicted_label == 1 else 'benign'
    fig, axes = plt.subplots(1, len(cam_images) + 1, figsize=(15, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image {image_id}')
    axes[0].axis('off')

    for i, cam_image in enumerate(cam_images):
        axes[i + 1].imshow(cam_image)
        axes[i + 1].set_title(f'Stage {stage + i} Fold {fold}\nPredicted Label: {label_text}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

def save_comparison(original_image, cam_images, image_id, fold, predicted_label, stage, output_file_path):
    label_text = 'malignant' if predicted_label == 1 else 'benign'
    fig, axes = plt.subplots(1, len(cam_images) + 1, figsize=(15, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image {image_id}')
    axes[0].axis('off')

    for i, cam_image in enumerate(cam_images):
        axes[i + 1].imshow(cam_image)
        axes[i + 1].set_title(f'Stage {stage + i} Fold {fold}\nPredicted Label: {label_text}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()

import random
if __name__ == "__main__":
    # Set a fixed seed for reproducibility
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
    args = config()
    # Ensure the mode is set to 'eval' for heatmap generation
    args.mode = 'eval'
    # Specific image names mentioned
    target_image_names = [
        "benign(53).png", "malignant(425).png", "malignant(938).png", "benign(191).png", "benign(433).png",
        "benign(441).png", "malignant(10).png", "malignant(100).png", "malignant(1083).png", "malignant(1090).png",
        "benign(164).png", "benign(20).png", "malignant(280).png", "malignant(786).png"
    ]

    # Base results folder
    results_folder = "/content/drive/MyDrive/weight_2024-08-20_23-49-44/GradCamPlusPlus"

    # Iterate over target images
    for image_name in target_image_names:
        img_path = os.path.join(images_folder, image_name)
        if os.path.exists(img_path):
            original_image = None

            # Create a folder for each image
            image_folder = os.path.join(results_folder, image_name)
            os.makedirs(image_folder, exist_ok=True)

            # Iterate over each fold
            for fold in range(5):
                # Load the model for the current fold
                model, img_size = load_model(fold)

                # Get target layers from all stages and the last stage
                target_layers_all_stages = [stage.merge.conv[-1] for stage in model.stage]
                target_layer_last_stage = model.stage[-1].merge.conv[-1]

                # Generate Grad-CAM for all stages
                original_image, cam_images_all_stages, predicted_label = generate_gradcam(
                    model, img_path, target_layers_all_stages, img_size
                )

                # Save and plot Grad-CAM for all stages
                output_file_path_all_stages = os.path.join(image_folder, f"fold_{fold}_gradcam_all_stages.png")
                save_comparison(original_image, cam_images_all_stages, image_name, fold, predicted_label, 0, output_file_path_all_stages)
                plot_comparison(original_image, cam_images_all_stages, image_name, fold, predicted_label, 0)

                # # Generate Grad-CAM for the last stage
                # original_image, cam_images_last_stage, predicted_label = generate_gradcam(
                #     model, img_path, [target_layer_last_stage], img_size
                # )

                # # Save and plot Grad-CAM for the last stage only
                # output_file_path_last_stage = os.path.join(image_folder, f"fold_{fold}_gradcam_last_stage.png")
                # save_comparison(original_image, cam_images_last_stage, image_name, fold, predicted_label, len(target_layers_all_stages), output_file_path_last_stage)
                # plot_comparison(original_image, cam_images_last_stage, image_name, fold, predicted_label, len(target_layers_all_stages))

                print(f"Processed {img_path} for fold {fold}, saved to {output_file_path_all_stages}")
        else:
            print(f"Image {img_path} not found.")


#GRADCAM for a model with 2 outputs,last stage

# import torch
# from torchvision import transforms
# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from hovertrans import create_model
# from config import config

# class ModelWrapper:
#     def __init__(self, model):
#         self.model = model

#     def eval(self):
#         self.model.eval()
#         return self

#     def __call__(self, x):
#         logits, _ = self.model(x)
#         return logits

#     def parameters(self):
#         return self.model.parameters()

#     def zero_grad(self):
#         self.model.zero_grad()

#     def to(self, device):
#         self.model.to(device)
#         return self

# def load_model():
#     args = config()
#     args.mode = 'eval'
#     model = create_model(
#         img_size=args.img_size,
#         num_classes=args.class_num,
#         drop_rate=0.1,
#         attn_drop_rate=0.1,
#         patch_size=args.patch_size,
#         dim=args.dim,
#         depth=args.depth,
#         num_heads=args.num_heads,
#         num_inner_head=args.num_inner_head
#     )
#     weight_path = f"/content/drive/MyDrive/bestmodel.pth"
#     model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
#     model.eval()
#     return ModelWrapper(model), args.img_size

# def generate_gradcam(model, img_path, target_layer, img_size):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.float32(cv2.resize(img, (img_size, img_size))) / 255.0
#     input_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.2706, 0.2671, 0.2874], std=[0.1761, 0.1746, 0.1856])
#     ])(img).unsqueeze(0)

#     cam = GradCAM(model=model, target_layers=[target_layer])
#     grayscale_cam = cam(input_tensor=input_tensor)[0]
#     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
#     # Print pixel values range
#     print("Min pixel value in Grad-CAM:", np.min(grayscale_cam))
#     print("Max pixel value in Grad-CAM:", np.max(grayscale_cam))

#     return img, cam_image

# def plot_comparison(original_image, cam_image, image_id):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     axes[0].imshow(original_image, cmap='gray')
#     axes[0].set_title(f'Original Image {image_id}')
#     axes[0].axis('off')

#     axes[1].imshow(cam_image)
#     axes[1].set_title(f'Grad-CAM {image_id}')
#     axes[1].axis('off')

#     plt.tight_layout()
#     plt.show()

# import random
# if __name__ == "__main__":
#     # Set a fixed seed for reproducibility
#     seed = 42
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = False

#     images_folder = "/content/drive/MyDrive/data/img"
#     args = config()
#     # Ensure the mode is set to 'eval' for heatmap generation
#     args.mode = 'eval'
#     # Specific image names mentioned
#     target_image_names = [
#         "benign(53).png", "malignant(425).png", "malignant(938).png", "bengin(191).png", "benign(433).png",
#         "benign(441).png", "malignant(10).png", "malignant(100).png", "malignant(1083).png", "malignant(1090).png",
#         "benign(164).png", "benign(20).png", "malignant(280).png", "malignant(786).png"
#     ]

#     # Base results folder
#     results_folder = "/content/drive/MyDrive/HoverTrans/heatmaps/ts128grad/1"
#     os.makedirs(results_folder, exist_ok=True)

#     model, img_size = load_model()

#     for image_name in target_image_names:
#         img_path = os.path.join(images_folder, image_name)
#         if os.path.exists(img_path):
#             original_image = None

#             # Only process the last stage
#             last_stage = model.model.stage[-1]
#             target_layer = last_stage.merge.conv[-1]
            
#             original_image, cam_image = generate_gradcam(model, img_path, target_layer, img_size)
                
#             image_folder = os.path.join(results_folder, image_name)
#             os.makedirs(image_folder, exist_ok=True)
#             plot_comparison(original_image, cam_image, f"{image_name}")

#             output_file_path = os.path.join(image_folder, f"{image_name}_gradcam.png")
#             fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#             axes[0].imshow(original_image, cmap='gray')
#             axes[0].set_title(f'Original Image {image_name}')
#             axes[0].axis('off')

#             axes[1].imshow(cam_image)
#             axes[1].set_title(f'Grad-CAM')
#             axes[1].axis('off')

#             plt.tight_layout()
#             plt.savefig(output_file_path)
#             plt.close()
#             print(f"Processed {img_path}, saved to {output_file_path}")
#         else:
#             print(f"Image {img_path} not found.")
