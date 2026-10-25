import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# ОПРЕДЕЛЕНИЕ МОДЕЛИ И ДАТАСЕТА

class DocumentEncoder(nn.Module):
    """Текстовый энкодер на базе HuggingFace Transformers."""

    def __init__(self, model_name="distilbert-base-uncased", embedding_dim=128):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # Линейный слой проекции в пространство эмбеддингов нужной размерности
        self.fc = nn.Linear(self.transformer.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Извлекаем вектор CLS-токена (первый токен последовательности)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings = self.fc(cls_embedding)
        # КРИТИЧЕСКИ ВАЖНО: CosFace требует строго L2-нормализованные векторы
        return nn.functional.normalize(embeddings, p=2, dim=1)


class TextDataset(Dataset):
    """Простой PyTorch Dataset для текстов."""

    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ПОДГОТОВКА ДАННЫХ И ИНИЦИАЛИЗАЦИЯ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Тестовый демонстрационный набор данных (3 класса документов)
train_texts = [
    "Tech update: Apple released a new iPhone with better battery",
    "Google announces breakthroughs in quantum computing architecture",
    "Real Madrid won the Champions League final yesterday",
    "The runner broke the world record in the 100m sprint",
    "Central bank raised interest rates to combat high inflation",
    "Stocks fell sharply after the corporate financial reports"
]
train_labels = [0, 0, 1, 1, 2, 2]  # 0: Tech, 1: Sports, 2: Finance

test_texts = [
    "Microsoft introduces a new AI assistant for software developers",
    "The football match ended in a draw after extra time",
    "Investors are buying gold amid global economic uncertainty"
]
test_labels = [0, 1, 2]

# Инициализация токенайзера, модели и даталоадеров
MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_DIM = 64
NUM_CLASSES = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DocumentEncoder(model_name=MODEL_NAME, embedding_dim=EMBEDDING_DIM).to(device)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# НАСТРОЙКА COSFACE LOSS И ОПТИМИЗАТОРА

# Инициализируем CosFaceLoss (внутри него обучаются центры классов)
loss_fn = losses.CosFaceLoss(
    num_classes=NUM_CLASSES,
    embedding_size=EMBEDDING_DIM,
    margin=0.35,
    scale=30
).to(device)

# Добавляем параметры лосса в общий оптимизатор
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 2e-5},
    {'params': loss_fn.parameters(), 'lr': 1e-3}
])

# ЦИКЛ ОБУЧЕНИЯ (FINE-TUNING С COSFACE)

print("Начало обучения модели...")
model.train()
for epoch in range(5):  # В реальной задаче ставьте больше эпох (например, 3-10)
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        embeddings = model(input_ids, attention_mask)
        loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Эпоха {epoch + 1}/5 | Средний Loss: {total_loss / len(train_loader):.4f}")


# ИЗВЛЕЧЕНИЕ ЭМБЕДДИНГОВ

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            embeddings = model(input_ids, attention_mask)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_embeddings), np.concatenate(all_labels)


print("\nИзвлечение признаков для k-NN...")
X_train, y_train = extract_embeddings(model, train_loader, device)
X_test, y_test = extract_embeddings(model, test_loader, device)

# КЛАССИФИКАЦИЯ С ПОМОЩЬЮ K-NN

print("Обучение KNeighborsClassifier...")
# Обязательно указываем metric='cosine'
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', weights='distance')
knn.fit(X_train, y_train)

# Делаем предсказания
y_pred = knn.predict(X_test)

# Выводим финальную аналитику
print("\n=== Результаты классификации документов ===")
target_names = ['Tech', 'Sports', 'Finance']
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
