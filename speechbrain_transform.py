import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

# 1. Load full classifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

# 2. Define end-to-end wrapper module
class ECAPAWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wavs):
        # wavs shape: (batch_size, samples) e.g., (1, 16000)
        feats = self.compute_features(wavs)
        
        # Apply length/mask normalization if needed
        lengths = torch.ones(feats.shape[0], device=wavs.device)
        feats = self.mean_var_norm(feats, lengths)
        
        # Pass 3D log-mel features into ECAPA-TDNN
        embeddings = self.embedding_model(feats)
        return embeddings

# 3. Export to ONNX using legacy tracer mode (dynamo=False avoids export graph issues)
model_to_export = ECAPAWrapper(classifier).eval()
dummy_wav = torch.randn(1, 32000) # 2 seconds of 16kHz audio

torch.onnx.export(
    model_to_export,
    dummy_wav,
    "ecapa_tdnn.onnx",
    input_names=["speech_wav"],
    output_names=["embedding"],
    dynamic_axes={
        "speech_wav": {1: "num_samples"},
        "embedding": {0: "batch_size"}
    },
    opset_version=17,
    dynamo=False  # Crucial to bypass PyTorch 2.x strict export tracing
)

print("Export successful!")