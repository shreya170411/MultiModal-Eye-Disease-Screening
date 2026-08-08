import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = r"E:\Major_Eye\modality_dataset"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

train_ds = datasets.ImageFolder(
    root=f"{data_dir}/train",
    transform=transform
)

val_ds = datasets.ImageFolder(
    root=f"{data_dir}/val",
    transform=transform
)

train_loader = DataLoader(train_ds,batch_size=16,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=16)

model = timm.create_model(
    "mobilenetv3_small_050",
    pretrained=True,
    num_classes=3
)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Classes:", train_ds.classes)

for epoch in range(5):

    model.train()

    total_loss = 0

    for img,label in train_loader:

        img,label = img.to(DEVICE),label.to(DEVICE)

        out = model(img)

        loss = criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch",epoch,"loss:",total_loss)

torch.save(
    model.state_dict(),
    r"E:\Major_Eye\modality\modality_model.pth"
)

print("Model saved.")