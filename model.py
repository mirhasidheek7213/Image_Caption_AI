import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# Define the ImageEncoder class using ResNet18
class ImageEncoder:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor).squeeze(-1).squeeze(-1)
        return features

# Define the Transformer_Decoder class for caption generation
class Transformer_Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 22, embed_size))  # Max_len=22
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=8,
            dim_feedforward=hidden_size,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions, mask=None):
        batch_size = captions.size(0)
        seq_len = captions.size(1)
        embedded = self.embedding(captions) + self.pos_encoding[:, :seq_len, :]
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(features.device)
        memory = features.unsqueeze(1)
        output = self.decoder(tgt=embedded, memory=memory, tgt_mask=mask)
        return self.fc(output)

# Define the CaptioningModel class to integrate encoder, decoder, and generation
class CaptioningModel:
    def __init__(self, model_path, vocab_path, embed_size=256, hidden_size=768, num_layers=6, device='cpu'):
        self.device = device
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.word2idx = vocab['word2idx']
        self.idx2word = vocab['idx2word']
        if isinstance(next(iter(self.idx2word)), str):
            self.idx2word = {int(k): v for k, v in self.idx2word.items()}

        self.encoder = ImageEncoder(device=device)
        self.decoder = Transformer_Decoder(embed_size=embed_size, vocab_size=len(self.word2idx), hidden_size=hidden_size, num_layers=num_layers).to(device)
        self.project_features = nn.Linear(512, embed_size).to(device)
        self._load_model(model_path)

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.project_features.load_state_dict(checkpoint['project_features'])
        self.decoder.eval()
        self.project_features.eval()

    def generate_caption(self, image, beam_width=5, max_len=22):
        image_features = self.encoder.encode(image)
        with torch.no_grad():
            projected = self.project_features(image_features)
        sequences = [[[self.word2idx["< SOS >"]], 0.0]]
        completed = []
        for _ in range(max_len):
            candidates = []
            for seq, score in sequences:
                if seq[-1] == self.word2idx["<EOS>"]:
                    completed.append((seq, score))
                    continue
                input_seq = torch.tensor([seq], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    output = self.decoder(projected, input_seq)
                    probs = torch.softmax(output[:, -1, :], dim=-1)
                    topk = torch.topk(probs, beam_width)
                for i in range(beam_width):
                    token = topk.indices[0, i].item()
                    token_prob = topk.values[0, i].item()
                    if token == self.word2idx["<UNK>"] and i < beam_width-1:
                        continue
                    new_seq = seq + [token]
                    new_score = score + torch.log(torch.tensor(token_prob + 1e-10)).item()
                    candidates.append((new_seq, new_score))
            sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if len(sequences) == 0:
                break
        all_sequences = sequences + completed
        if not all_sequences:
            return ""
        best_seq = sorted(all_sequences, key=lambda x: x[1], reverse=True)[0][0]
        caption_tokens = [self.idx2word[t] for t in best_seq if t not in [self.word2idx["<PAD>"], self.word2idx["< SOS >"], self.word2idx["<EOS>"]]]
        return " ".join(caption_tokens)

# Function to load the model and generate a caption
def load_model_and_generate_caption(image_path, model_path, vocab_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CaptioningModel(model_path=model_path, vocab_path=vocab_path, device=device)
    image = Image.open(image_path).convert('RGB')
    caption = model.generate_caption(image)
    return caption