from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

class MiMoConfig(Qwen2Config):
    model_type = "mimo"

    def __init__(
        self,
        *args,
        num_nextn_predict_layers=0,
        **kwargs
    ):
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(
            *args,
            **kwargs,
        )
