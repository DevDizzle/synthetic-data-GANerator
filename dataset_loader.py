from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Define the Dataset
class MalignantDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = list(Path(image_folder).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Data transformations
def get_dataloader(image_folder, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    dataset = MalignantDataset(image_folder, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
