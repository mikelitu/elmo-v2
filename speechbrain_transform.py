import torch
import torch.nn as nn
from speechbrain.inference.speaker import SpeakerRecognition

# Load pre-trained SpeechBrain model
sb_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_sb_model"
)

class ECAPAEncoderOnly(nn.Module):
    """Wraps Normalization + ECAPA_TDNN to bypass STFT ONNX tracing errors."""
    def __init__(self, mods):
        super().__init__()
        self.mean_var_norm = mods.mean_var_norm
        self.embedding_model = mods.embedding_model

    def forward(self, feats):
        # feats shape: [batch_size, time_frames, num_mels (80)]
        lengths = torch.ones(feats.shape[0], device=feats.device)
        norm_feats = self.mean_var_norm(feats, lengths)
        embeddings = self.embedding_model(norm_feats)
        return embeddings

# Prepare wrapper
encoder_wrapper = ECAPAEncoderOnly(sb_model.mods)
encoder_wrapper.eval()

# Dummy input representing Mel-Filterbank Features [Batch=1, Frames=301, Mel_Bins=80]
dummy_feats = torch.randn(1, 301, 80, dtype=torch.float32)

onnx_filename = "ecapa_tdnn_encoder.onnx"

torch.onnx.export(
    encoder_wrapper,
    dummy_feats,
    onnx_filename,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['mel_features'],
    output_names=['embedding'],
    dynamic_axes={
        'mel_features': {0: 'batch_size', 1: 'time_frames'},
        'embedding': {0: 'batch_size'}
    }
)

print(f"ONNX Encoder model successfully exported to {onnx_filename}")