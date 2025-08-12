from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir : Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int 
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_dir: Path
    validation_dir: Path
    params_epochs: int
    params_batch_size: int 
    params_is_augmentation: bool
    params_image_size: list
    params_warmup_epochs:int
    params_unfreeze_layers: int 
    params_finetune_lr: float
    params_random_state: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model:Path
    test_dir: Path
    all_params:dict
    mlflow_uri:str
    params_image_size:list
    params_batch_size:int