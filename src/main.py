from datetime import datetime
from typing import Optional
import os

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import torch.optim as optim
import wandb
import argparse
# ModelCheckpoint für das Speichern des Modells
from pytorch_lightning.callbacks import ModelCheckpoint

# Hyperparameter fix
adam_epsilon = 1e-8
weight_decay = 0
train_batch_size_value: int = 32 # Für Docker Playground wurde dieser Wert auf 8 gesetzt, um die Laufzeit zu verkürzen. Original: 32
eval_batch_size_value: int = 32 # Für Docker Playground wurde dieser Wert auf 8 gesetzt, um die Laufzeit zu verkürzen. Original: 32
max_epochs_trainer = 3 # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Original: 3
# num_workers = 1 # Für Docker Playground wurde dieser Wert auf 0 gesetzt, um die Laufzeit zu verkürzen. Wurde lokale Verwendung wieder deaktivert

# Parameterübergabe über Kommandozeile
parser = argparse.ArgumentParser(description="Training script for MLOps project")
# Neues Argument für den API-Schlüssel
parser.add_argument("--api_key", type=str, default=os.environ.get("API_KEY"), help="WandB API key")
parser.add_argument("--wandb_projectname", type=str, default="MLOPS_Project2", help="WandB project name")
parser.add_argument("--save_path", type=str, default="./checkpoint", help="Save path for trained model")
parser.add_argument("--learning_rate", type=float, default=float(os.environ.get("LEARNING_RATE", 2.84468e-5)), help="Learning rate")
parser.add_argument("--warmup_steps", type=float, default=float(os.environ.get("WARMUP_STEPS", 209.9549193)), help="Warmup steps")
parser.add_argument("--optimizer_choice", type=str, default=os.environ.get("OPTIMIZER_CHOICE", "adam"), choices=["adam", "sgd", "rmsprop"], help="Optimizer choice")
args = parser.parse_args()

# api_key = os.environ.get("API_KEY")

# Parameter aus Kommandozeile auslesen und in Variablen speichern:
wandb_projectname = args.wandb_projectname
save_path = args.save_path
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps
optimizer_choice = args.optimizer_choice
api_key = args.api_key

if api_key:
    wandb.login(key=api_key)
else:
    raise ValueError("API key not provided. Please set the API_KEY environment variable.")

class GLUEDataModule(LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = train_batch_size_value, # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Original: 32
        eval_batch_size: int = eval_batch_size_value, # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Original: 32
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size_value
        self.eval_batch_size = eval_batch_size_value

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True) # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Optional: num_workers=num_workers

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size) # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Optional: num_workers=num_workers
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits] # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop. Optional: num_workers=num_workers

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    
class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = learning_rate,
        adam_epsilon: float = adam_epsilon,
        warmup_steps: int = warmup_steps,
        weight_decay: float = weight_decay,
        train_batch_size: int = train_batch_size_value, # train_batch_size,
        eval_batch_size: int = eval_batch_size_value, # eval_batch_size,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        # wandb.log(self.hparams)

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        # Protokollieren des Verlusts in wandb
        self.log("train_loss", loss.item())
        #
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
                # Protokollieren des Validierungsverlusts und Metriken in wandb
                wandb.log({f"val_loss_{split}": loss.item(), **split_metrics})
                #
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        # Protokollieren des Validierungsverlusts und Metriken in wandb
        wandb.log({"val_loss": loss.item(), **self.metric.compute(predictions=preds, references=labels)})
        #

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if optimizer_choice == "adam":
          optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        elif optimizer_choice == "sgd":
          optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        elif optimizer_choice == "rmsprop":
            optimizer = optim.RMSprop(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        elif optimizer_choice == "adagrad":
            optimizer = optim.Adagrad(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        else:
          raise ValueError("Ungültiger Optimizer-Typ. Unterstützte Typen sind 'adam' und 'sgd'.")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        for i, param_group in enumerate(optimizer.param_groups):
          wandb.log({f"lr_group_{i}": param_group["lr"]})


seed_everything(42)

# Hyperparameter in einen Namen für den Run einfügen
run_name = f"lr_{learning_rate}_warmup_{warmup_steps}_opt_{optimizer_choice}"

run = wandb.init(
    project=wandb_projectname,
    #group="week2",
    # Name wird anhand der Hyperparameterwahl erstellt
    name= run_name,
    config={
        "learning_rate": learning_rate,
        "adam_epsilon": adam_epsilon,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "train_batch_size": train_batch_size_value, # train_batch_size,
        "eval_batch_size": eval_batch_size_value, # eval_batch_size,
        "optimizer_choice": optimizer_choice,
    })

dm = GLUEDataModule(
    model_name_or_path="distilbert-base-uncased",
    task_name="mrpc",
)
dm.setup("fit")
model = GLUETransformer(
    model_name_or_path="distilbert-base-uncased",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
)


checkpoint_callback = ModelCheckpoint(
    dirpath=save_path,  # Speicherort von Checkpoints, wird über Kommandozeile übergeben
    filename="best_model",
    save_top_k=1,  # nur den besten Model-Checkpoint
    verbose=True,
    monitor="val_loss",
    mode="min",
)


trainer = Trainer(
    max_epochs=max_epochs_trainer, # Anpassung, um die Laufzeit zu verkürzen auf meinem Laptop, Original: 3
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    callbacks=[checkpoint_callback],  # Hinzufügen des Checkpoint-Callbacks
)
trainer.fit(model, datamodule=dm)


wandb.finish()