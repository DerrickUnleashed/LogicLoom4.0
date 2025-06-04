import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from huggingface_hub import login
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class HazardClassifier:
    def __init__(self):
        """Initialize the hazard classifier with configurations."""
        self.setup_logging()
        self.load_environment()
        self.setup_device()
        self.setup_labels()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_environment(self):
        """Load environment variables and authenticate."""
        load_dotenv()
        
        # HuggingFace authentication
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if hf_token:
            try:
                login(hf_token)
                self.logger.info("Successfully authenticated with HuggingFace")
            except Exception as e:
                self.logger.warning(f"HuggingFace authentication failed: {e}")
        
        # WandB setup (optional)
        wandb_key = os.environ.get('WANDB_API_KEY')
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            self.logger.info("WandB API key configured")
        else:
            # Disable wandb if not configured
            os.environ["WANDB_DISABLED"] = "true"
            
    def setup_device(self):
        """Setup the appropriate device for training."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using MPS (Apple Silicon) device")
            # MPS-specific optimizations
            # Clear MPS cache
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU device")
    
    def setup_training_for_mps(self, training_args_dict):
        """Apply MPS-specific training optimizations."""
        if self.device.type == "mps":
            # MPS works better with specific settings
            training_args_dict.update({
                "dataloader_num_workers": 0,  # MPS doesn't work well with multiprocessing
                "fp16": False,  # MPS doesn't support fp16, use default precision
                "dataloader_pin_memory": False,  # Disable pin memory for MPS
            })
            self.logger.info("Applied MPS-specific optimizations")
        return training_args_dict
            
    def setup_labels(self):
        """Define label mappings."""
        self.label2id = {
            "allergens": 0,
            "biological": 1,
            "chemical": 2,
            "food additives and flavourings": 3,
            "foreign bodies": 4,
            "fraud": 5,
            "migration": 6,
            "organoleptic aspects": 7,
            "other hazard": 8,
            "packaging defect": 9,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.logger.info(f"Configured {len(self.label2id)} hazard categories")
        
    def load_and_prepare_data(self, train_file="Hazards_LABELLED_TRAIN.csv"):
        """Load and prepare training data."""
        try:
            if not Path(train_file).exists():
                raise FileNotFoundError(f"Training file {train_file} not found")
                
            df = pd.read_csv(train_file)
            self.logger.info(f"Loaded {len(df)} training samples")
            
            # Validate required columns
            required_cols = ['text', 'hazard-type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and prepare data
            df = df.dropna(subset=required_cols)
            df['label'] = df['hazard-type'].map(self.label2id)
            
            # Handle unmapped labels
            unmapped = df['label'].isna().sum()
            if unmapped > 0:
                self.logger.warning(f"Found {unmapped} samples with unmapped hazard types")
                df = df.dropna(subset=['label'])
                
            df['input'] = df['text'].astype(str)
            
            # Ensure labels are integers (required for classification)
            df['label'] = df['label'].astype(int)
            
            df = df[['input', 'label']].reset_index(drop=True)
            
            self.logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
            
            # Split data
            train_df, val_df = train_test_split(
                df, test_size=0.15, random_state=42, stratify=df['label']
            )
            
            self.logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
            
            # Create datasets
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            self.dataset = DatasetDict({
                'train': train_dataset, 
                'validation': val_dataset
            })
            
            # Verify dataset structure
            self.logger.info(f"Dataset features: {self.dataset['train'].features}")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def setup_model_and_tokenizer(self, model_checkpoint='distilbert-base-uncased'):
        """Setup model and tokenizer with proper configuration."""
        try:
            self.logger.info(f"Loading model: {model_checkpoint}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            
            # Handle padding token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                problem_type="single_label_classification"
            )
            
            # Resize embeddings if needed
            if len(self.tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            # Move model to device
            self.model.to(self.device)
            
            self.logger.info("Model and tokenizer setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise
            
    def setup_peft(self, use_peft=True):
        """Setup PEFT (Parameter Efficient Fine-Tuning) if requested."""
        if not use_peft:
            return
            
        try:
            # Configure LoRA based on model type
            model_name = self.model.config.model_type.lower()
            
            if 'gpt2' in model_name:
                target_modules = ["c_attn", "c_proj"]
            elif 'bert' in model_name:
                # For BERT models, check actual layer names
                target_modules = []
                for name, _ in self.model.named_modules():
                    if any(layer in name for layer in ["query", "value", "key", "dense"]):
                        target_modules.append(name.split('.')[-1])
                        break
                # Fallback to common BERT patterns
                if not target_modules:
                    target_modules = ["query", "value"]
            elif 'distilbert' in model_name:
                # DistilBERT specific modules
                target_modules = ["q_lin", "v_lin", "k_lin", "out_lin"]
            else:
                # Generic transformer modules
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
                
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=target_modules,
                bias="none"
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.logger.info(f"PEFT configuration applied with target modules: {target_modules}")
            
        except Exception as e:
            self.logger.warning(f"PEFT setup failed, continuing without: {e}")
            # Print available modules for debugging
            self.logger.info("Available modules in the model:")
            for name, _ in self.model.named_modules():
                if any(keyword in name.lower() for keyword in ['attention', 'dense', 'linear', 'proj']):
                    self.logger.info(f"  - {name}")
            
    def tokenize_dataset(self):
        """Tokenize the dataset."""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["input"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors=None
            )
            # Keep the labels - this is crucial for training
            tokenized["labels"] = examples["label"]
            return tokenized
        
        # Remove only the text columns, keep labels
        columns_to_remove = [col for col in self.dataset["train"].column_names if col not in ["label"]]
        
        self.tokenized_dataset = self.dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=columns_to_remove
        )
        
        self.logger.info("Dataset tokenization completed")
        self.logger.info(f"Tokenized dataset columns: {self.tokenized_dataset['train'].column_names}")
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        
        acc_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        
        return {
            "accuracy": acc_score,
            "f1": f1_score
        }
        
    def train_model(self, output_dir="hazard-classifier", use_early_stopping=True, **training_kwargs):
        """Train the model with optimized arguments."""
        
        # Auto-detect supported TrainingArguments parameters
        import inspect
        training_args_signature = inspect.signature(TrainingArguments.__init__)
        supported_params = set(training_args_signature.parameters.keys())
        
        # Base training arguments
        base_args = {
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16 if self.device.type == "cuda" else 8,
            "per_device_eval_batch_size": 16 if self.device.type == "cuda" else 8,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "logging_steps": 50,
            "eval_steps": 200,
            "save_steps": 200,
            "dataloader_num_workers": 0,
            "fp16": self.device.type == "cuda",  # Only use fp16 on CUDA
        }
        
        # Apply MPS-specific optimizations
        base_args = self.setup_training_for_mps(base_args)
        
        # Add evaluation strategy and early stopping requirements
        if "evaluation_strategy" in supported_params:
            base_args.update({
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
            })
            
            # Only add early stopping requirements if we're using early stopping
            if use_early_stopping:
                base_args.update({
                    "metric_for_best_model": "f1",
                    "greater_is_better": True,
                })
        elif "do_eval" in supported_params:
            base_args.update({
                "do_eval": True,
                "save_steps": 200,
            })
        
        # Add reporting (WandB)
        if "report_to" in supported_params:
            base_args["report_to"] = None if os.environ.get("WANDB_DISABLED") else "wandb"
        
        # Update with user provided arguments
        base_args.update(training_kwargs)
        
        # Filter out unsupported parameters
        filtered_args = {k: v for k, v in base_args.items() if k in supported_params}
        
        self.logger.info(f"Using training arguments: {list(filtered_args.keys())}")
        
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                **filtered_args
            )
        except Exception as e:
            self.logger.error(f"Failed to create TrainingArguments with filtered params: {e}")
            # Fallback to minimal arguments
            minimal_args = {
                "output_dir": output_dir,
                "learning_rate": base_args["learning_rate"],
                "per_device_train_batch_size": base_args["per_device_train_batch_size"],
                "num_train_epochs": base_args["num_train_epochs"],
                "logging_steps": base_args["logging_steps"],
            }
            training_args = TrainingArguments(**minimal_args)
            use_early_stopping = False  # Disable early stopping for fallback
            self.logger.warning("Using minimal training arguments due to compatibility issues")
        
        # Setup trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": self.tokenized_dataset["train"],
            "tokenizer": self.tokenizer,
            "compute_metrics": self.compute_metrics,
        }
        
        # Add eval dataset if evaluation is configured
        if hasattr(training_args, 'evaluation_strategy') and training_args.evaluation_strategy != "no":
            trainer_kwargs["eval_dataset"] = self.tokenized_dataset["validation"]
        elif hasattr(training_args, 'do_eval') and training_args.do_eval:
            trainer_kwargs["eval_dataset"] = self.tokenized_dataset["validation"]
        
        # Add callbacks if supported and requested
        callbacks = []
        if use_early_stopping and hasattr(training_args, 'metric_for_best_model') and training_args.metric_for_best_model:
            try:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
                self.logger.info("Added EarlyStoppingCallback")
            except Exception as e:
                self.logger.warning(f"Could not add EarlyStoppingCallback: {e}")
        
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks
        
        self.trainer = Trainer(**trainer_kwargs)
        
        self.logger.info("Starting training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"Training completed. Model saved to {output_dir}")
            self.logger.info(f"Training metrics: {train_result.metrics}")
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
    def predict_single(self, text):
        """Predict hazard type for a single text."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            "predicted_class": self.id2label[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                self.id2label[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
    def predict_batch(self, test_file="Hazards_UNLABELLED_TEST.csv", output_file="predictions.csv"):
        """Make predictions on test data."""
        try:
            if not Path(test_file).exists():
                self.logger.warning(f"Test file {test_file} not found, skipping batch prediction")
                return None
                
            df_test = pd.read_csv(test_file)
            self.logger.info(f"Loaded {len(df_test)} test samples")
            
            # Prepare test dataset
            df_test['input'] = df_test['text'].astype(str)
            test_dataset = Dataset.from_pandas(df_test[['input']])
            
            def tokenize_test(examples):
                return self.tokenizer(
                    examples["input"],
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
            
            tokenized_test = test_dataset.map(tokenize_test, batched=True)
            
            # Make predictions
            predictions = self.trainer.predict(tokenized_test)
            predicted_labels = np.argmax(predictions.predictions, axis=1)
            predicted_classes = [self.id2label[pred] for pred in predicted_labels]
            
            # Add predictions to dataframe
            df_test['predicted_hazard_type'] = predicted_classes
            
            # Save results
            if 'ID' in df_test.columns:
                df_test[['ID', 'predicted_hazard_type']].to_csv(output_file, index=False)
            else:
                df_test[['predicted_hazard_type']].to_csv(output_file, index=False)
                
            self.logger.info(f"Predictions saved to {output_file}")
            
            return df_test
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
            
    def evaluate_model(self, test_data=None):
        """Evaluate model performance."""
        if test_data is None:
            # Use validation set
            eval_results = self.trainer.evaluate()
            self.logger.info(f"Validation results: {eval_results}")
            return eval_results
        else:
            # Evaluate on provided test data with labels
            if 'hazard-type' in test_data.columns:
                true_labels = test_data['hazard-type'].values
                pred_labels = test_data['predicted_hazard_type'].values
                
                report = classification_report(
                    true_labels, pred_labels, 
                    target_names=list(self.label2id.keys()),
                    output_dict=True
                )
                
                self.logger.info("Classification Report:")
                print(classification_report(true_labels, pred_labels, target_names=list(self.label2id.keys())))
                
                return report
                
def main():
    """Main training pipeline."""
    classifier = HazardClassifier()
    
    try:
        # Load and prepare data
        dataset = classifier.load_and_prepare_data()
        
        # Setup model (using DistilBERT for better performance and speed)
        classifier.setup_model_and_tokenizer('distilbert-base-uncased')
        
        # Setup PEFT (optional - set to False for full fine-tuning)
        classifier.setup_peft(use_peft=True)
        
        # Tokenize dataset
        classifier.tokenize_dataset()
        
        # Train model
        train_result = classifier.train_model(
            output_dir="hazard-classifier-final",
            num_train_epochs=3,
            learning_rate=2e-5,
            use_early_stopping=True  # Set to False to disable early stopping
        )
        
        # Make predictions on test data
        test_results = classifier.predict_batch()
        
        # Evaluate if test labels are available
        if test_results is not None:
            classifier.evaluate_model(test_results)
            
        classifier.logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        classifier.logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()