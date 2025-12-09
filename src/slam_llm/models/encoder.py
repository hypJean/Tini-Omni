import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperWrappedEncoder:

    @classmethod
    def load(cls, model_config):

        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        if getattr(model_config, "whisper_decode", False):
            import whisper

            whisper_model = whisper.load_model(
                name=model_config.encoder_path, device="cpu"
            )
            whisper_model.encoder.extract_variable_length_features = types.MethodType(
                extract_variable_length_features, whisper_model.encoder
            )
            return whisper_model

        if getattr(model_config, "encoder_path_hf", None) is not None:
            from transformers import WhisperModel

            encoder = WhisperModel.from_pretrained(
                model_config.encoder_path_hf, torch_dtype=torch.bfloat16
            ).encoder
        else:
            import whisper

            encoder = whisper.load_model(
                name=model_config.encoder_path, device="cpu"
            ).encoder
            encoder.extract_variable_length_features = types.MethodType(
                extract_variable_length_features, encoder
            )
        return encoder


class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model