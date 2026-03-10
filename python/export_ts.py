import torch
from train_cnn_1class_v2 import COPDClassifier, NUM_CLASSES, IMG_SIZE

def main():
    model_path = "/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2/best_model_fold_0.pth"
    output_path = "/home/iec/Parallel_Computing_on_FPGA/python/output_copd_v2/best_model_fold_0.pt"

    print(f"Loading model from {model_path}...")
    model = COPDClassifier(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"Tracing model with dummy input of size (1, 3, {IMG_SIZE}, {IMG_SIZE})...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save the traced model
    traced_model.save(output_path)
    print(f"TorchScript model successfully saved to {output_path}")

if __name__ == "__main__":
    main()
