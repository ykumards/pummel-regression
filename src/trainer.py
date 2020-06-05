import os
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
from pathlib import Path

import dataset, vocab, utils
import models.ehr_lstm as model


class Trainer(object):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        model: nn.Module,
        args: dict,
        writer=None,
    ):
        self.args = args
        self.writer = writer

        exp_name = args.exp_name
        exp_dir = args.exp_dir / exp_name
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)

        self.model = model
        self.args = args
        self.dataset = dataset
        self.model = model.to(device=args.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=True,
        )

        self.train_state = {
            "done_training": False,
            "stop_early": False,
            "es_step": 0,
            "reduce_lr_step": 0,
            "es_best_val": float("inf"),
            "es_criteria": args.es_criteria,
            "reduce_lr_criteria": args.reduce_lr_criteria,
            "learning_rate": args.learning_rate,
            "epoch_index": 0,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_true": [],
            "val_preds": [],
            "test_loss": -1,
            "test_acc": 0,
            "test_preds": [],
            "test_true": [],
            "heldout_loss": -1,
            "heldout_acc": 0,
            "best_val_loss": -1,
            "best_tr_loss": -1,
            "heldout_preds": [],
            "heldout_true_preds_id_tuple": [],
            "heldout_true": [],
            "optim_filename": args.optim_state_file,
            "model_filename": args.model_state_file,
        }

    def compute_accuracy(
        self, y_pred: torch.FloatTensor, y_target: torch.FloatTensor
    ) -> float:
        "Acccuracy metric for this regression problem is R2"
        y_pred = y_pred.cpu().detach().numpy()
        y_target = y_target.cpu().detach().numpy()
        return metrics.r2_score(y_target, y_pred)

    def pad_x_sequence_along_diagnoses(
        self, seq: list, max_diagnosis_length: int, padding_value: int = 0
    ) -> torch.LongTensor:
        """
        Args: 
          seq: list of lists [num_visits, max_diag_length_per_visit]
          max_seq_length: max length of sequence across the batches
          ax_diagnosis_length: max_diag_ln_across_batches

        Returns:
          padded_vector: torch Tensor [num_visits, max_diagnosis_length] padded by 0
        """
        padded_vector_along_diagnoses = np.zeros((len(seq), max_diagnosis_length))
        for i, visit in enumerate(seq):
            padded_vector_along_diagnoses[
                i, : len(visit)
            ] = visit  # pads zeros to the front

        return torch.LongTensor(padded_vector_along_diagnoses)

    def collate_fn(self, batch):
        """
        Handles the padding of the input sequences. It handles padding 
        both in the num_visits and max_diagnosis_length dimensions 

        Args:
          batch: list of dictionary with keys 'x_seq', 'target_y', 'seq_length'
                 x_seq is ndarray [num_visits, max_diagnosis_length]

        Returns:
          processed_batch: dictionary with same keys containing list of values, 
                           but the arrays are Tensors padded according to the 
                           largest seq in the batch 
                           of shape [bs, max_num_visits_in_batch, ...]
        """
        max_seq_length = self.args.max_seq_length

        processed_batch = {
            "x_seq": [],
            "target_y": [],
            "seq_length": [],
            "age": [],
            "gender": [],
            "time_delta_mean": [],
            "prev_num_visits": [],
            "timedels": [],
            "krypht": [],
        }

        for _, sample in enumerate(batch):
            padded_x_seq_along_diagnoses = self.pad_x_sequence_along_diagnoses(
                sample["x_seq"], self.args.max_diag_length_per_visit
            )

            processed_batch["x_seq"].append(padded_x_seq_along_diagnoses)
            processed_batch["seq_length"].append(sample["seq_length"])
            processed_batch["target_y"].append(sample["target_y"])

            processed_batch["timedels"].append(torch.FloatTensor(sample["timedels"]))
            processed_batch["age"].append(sample["age"])
            processed_batch["gender"].append(sample["gender"])
            processed_batch["krypht"].append(sample["krypht"])
            processed_batch["prev_num_visits"].append(sample["prev_num_visits"])
            processed_batch["time_delta_mean"].append(sample["time_delta_mean"])

        processed_batch["x_seq"] = nn.utils.rnn.pad_sequence(
            processed_batch["x_seq"], batch_first=True
        )
        processed_batch["seq_length"] = torch.LongTensor(processed_batch["seq_length"])
        processed_batch["target_y"] = torch.LongTensor(
            processed_batch["target_y"]
        ).squeeze(0)

        if self.args.include_timedels:
            processed_batch["timedels"] = nn.utils.rnn.pad_sequence(
                processed_batch["timedels"], batch_first=True
            )

        if self.args.include_features:
            processed_batch["age"] = torch.FloatTensor(
                processed_batch["age"]
            ).unsqueeze(1)
            processed_batch["prev_num_visits"] = torch.FloatTensor(
                processed_batch["prev_num_visits"]
            ).unsqueeze(1)
            processed_batch["time_delta_mean"] = torch.FloatTensor(
                processed_batch["time_delta_mean"]
            ).unsqueeze(1)
            processed_batch["gender"] = torch.LongTensor(processed_batch["gender"])

        return processed_batch

    def update_train_state(self, verbose=True):
        ep = self.train_state["epoch_index"]
        lr = self.train_state["learning_rate"]
        tr_loss = self.train_state["train_loss"][-1]
        tr_acc = self.train_state["train_acc"][-1]
        val_loss = self.train_state["val_loss"][-1]
        val_acc = self.train_state["val_acc"][-1]
        curr_patience = self.train_state["es_step"]

        if self.writer is not None:
            self.writer.add_scalar("train_loss", tr_loss, ep)
            self.writer.add_scalar("train_acc", tr_acc, ep)
            self.writer.add_scalar("validation_loss", val_loss, ep)
            self.writer.add_scalar("validation_acc", val_acc, ep)
            self.writer.add_histogram(
                "validation y_preds", np.array(self.train_state["val_preds"]), ep
            )
            self.writer.add_histogram(
                "validation y_true", np.array(self.train_state["val_true"]), ep
            )

        if self.train_state["epoch_index"] == 0:
            torch.save(self.model.state_dict(), self.train_state["model_filename"])
            torch.save(self.optimizer.state_dict(), self.train_state["optim_filename"])
            self.train_state["stop_early"] = False

        else:
            loss_t = self.train_state["val_loss"][-1]
            if loss_t >= self.train_state["es_best_val"] + self.args.es_threshold:
                self.train_state["es_step"] += 1
                self.train_state["reduce_lr_step"] += 1
            else:
                # save model only if loss is absolutely less than previous best
                # dont consider threshold
                if loss_t <= self.train_state["es_best_val"]:
                    print("best model, saving...")

                    self.train_state["es_best_val"] = loss_t
                    self.train_state["best_val_loss"] = loss_t
                    self.train_state["best_tr_loss"] = tr_loss

                    torch.save(
                        self.model.state_dict(), self.train_state["model_filename"]
                    )
                    torch.save(
                        self.optimizer.state_dict(), self.train_state["optim_filename"]
                    )

                # Reset early stopping step with threshold
                self.train_state["es_step"] = 0
                self.train_state["reduce_lr_step"] = 0

            if (self.train_state["epoch_index"] == 5) or (
                self.train_state["reduce_lr_step"]
                >= self.train_state["reduce_lr_criteria"]
            ):
                print("es hit, reducing LR...")
                self.model.load_state_dict(
                    torch.load(self.train_state["model_filename"])
                )
                self.optimizer.load_state_dict(
                    torch.load(self.train_state["optim_filename"])
                )
                self.train_state["learning_rate"] *= 0.1

                for g in self.optimizer.param_groups:
                    g["lr"] = self.train_state["learning_rate"]
                self.train_state["reduce_lr_step"] = 0

            if self.train_state["es_step"] >= self.train_state["es_criteria"]:
                print("stopping early...")
                self.train_state["stop_early"] = True

        print_lr = 0.0
        for param_group in self.optimizer.param_groups:
            print_lr = param_group["lr"]

        # Verbose
        if verbose:
            print(
                f"[{ep}]: | lr: {print_lr} | tr_loss: {tr_loss:.4f} | "
                + f"tr_acc: {tr_acc:.2f} | val_loss: {val_loss:.4f} | "
                + f"val_acc: {val_acc:.2f} | patience: {curr_patience}"
            )

        return self.train_state

    def tune_lr(self, how_many_trys=100):
        for i in range(how_many_trys):
            lr = 10 ** np.random.uniform(-6, -1)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            print("=" * 90)
            print(f"[{i+1}/{how_many_trys}] | [lr]: {lr}")
            self.train(verbose=False)
            print("HO loss: {0:.2f}".format(self.train_state["heldout_loss"]))
            print("HO Accuracy: {0:.1f}%".format(self.train_state["heldout_acc"]))

    def _get_model_addln_args(self, batch_dict):

        model_addln_args = {"all_input_zeros": self.args.all_input_zeros}

        if self.args.include_timedels and batch_dict["timedels"] is not None:
            model_addln_args["x_timedels"] = batch_dict["timedels"].to(self.args.device)

        if self.args.include_features and batch_dict["age"] is not None:
            model_addln_args["x_age"] = batch_dict["age"].to(self.args.device)

        if self.args.include_features and batch_dict["gender"] is not None:
            model_addln_args["x_gender"] = batch_dict["gender"].to(self.args.device)

        if self.args.include_features:
            model_addln_args["x_time_delta_mean"] = batch_dict["time_delta_mean"].to(
                self.args.device
            )

        return model_addln_args

    def train(self, verbose=True):
        if verbose:
            print(f"begin training, stopping at patience: {self.args.es_criteria}")
            n_params = sum([np.prod(p.size()) for p in list(self.model.parameters())])
            print(self.model)
            print("=" * 90)
            print(f"model has {n_params} parameters")

        for epoch in range(self.args.num_epochs):
            self.train_state["epoch_index"] = epoch
            self.dataset.set_split("train")
            batch_generator = self.dataset.generate_batches(
                batch_size=self.args.batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.args.shuffle,
                device=self.args.device,
            )

            running_loss = 0.0
            running_acc = 0.0

            self.model.train()

            for batch_idx, batch_dict in enumerate(
                tqdm(
                    batch_generator,
                    total=self.dataset.get_num_batches(self.args.batch_size),
                )
            ):
                self.optimizer.zero_grad()
                ## get all the arguments together
                model_addln_args = self._get_model_addln_args(batch_dict)
                y_pred = self.model(
                    batch_dict["x_seq"].to(self.args.device),
                    batch_dict["seq_length"].to(self.args.device),
                    **model_addln_args,
                )

                loss = self.loss_func(
                    y_pred, batch_dict["target_y"].to(self.args.device).float()
                )

                running_loss += loss.item() * len(batch_dict["x_seq"])  # batchsize
                acc_t = self.compute_accuracy(y_pred, batch_dict["target_y"])
                running_acc += acc_t * len(batch_dict["x_seq"])

                loss.backward()
                self.optimizer.step()

            self.train_state["train_loss"].append(running_loss / len(self.dataset))
            self.train_state["train_acc"].append(running_acc / len(self.dataset))

            ##### Validation Steps ##############
            self.dataset.set_split("val")
            batch_generator = self.dataset.generate_batches(
                batch_size=self.args.batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.args.shuffle,
                device=self.args.device,
            )

            self.model.eval()
            val_running_loss = 0.0
            val_running_acc = 0.0

            for batch_dict in tqdm(
                batch_generator,
                total=self.dataset.get_num_batches(self.args.batch_size),
            ):
                # compute the output (forward pass)
                model_addln_args = self._get_model_addln_args(batch_dict)

                y_pred = self.model(
                    batch_dict["x_seq"].to(self.args.device),
                    batch_dict["seq_length"].to(self.args.device),
                    **model_addln_args,
                )

                loss = self.loss_func(
                    y_pred, batch_dict["target_y"].to(self.args.device).float()
                )
                val_running_loss += loss.item() * len(batch_dict["x_seq"])
                acc_t = self.compute_accuracy(y_pred, batch_dict["target_y"])
                val_running_acc += acc_t * len(batch_dict["x_seq"])
                self.train_state["val_preds"] += y_pred.tolist()
                self.train_state["val_true"] += batch_dict["target_y"].tolist()

            self.train_state["val_loss"].append(val_running_loss / len(self.dataset))
            self.train_state["val_acc"].append(val_running_acc / len(self.dataset))

            self.train_state = self.update_train_state(verbose)
            self.plot_performance(filename=f"{self.args.exp_name}_training.png")
            if self.train_state["stop_early"]:
                break

        self.heldout()

    def heldout(self):
        ## Report final validation loss
        self.dataset.set_split("heldout")
        batch_generator = self.dataset.generate_batches(
            batch_size=self.args.batch_size * 2,
            collate_fn=self.collate_fn,
            shuffle=False,
            device=self.args.device,
        )

        ## Load the best model so far
        self.model.load_state_dict(torch.load(self.train_state["model_filename"]))
        self.model.eval()

        ho_running_loss = 0.0
        ho_running_acc = 0.0

        for batch_dict in tqdm(
            batch_generator, total=self.dataset.get_num_batches(self.args.batch_size)
        ):
            # compute the output (forward pass)
            model_addln_args = self._get_model_addln_args(batch_dict)

            y_pred = self.model(
                batch_dict["x_seq"].to(self.args.device),
                batch_dict["seq_length"].to(self.args.device),
                **model_addln_args,
            )

            # compute the loss
            loss = self.loss_func(
                y_pred, batch_dict["target_y"].to(self.args.device).float()
            )

            ho_running_loss += loss.item() * len(batch_dict["x_seq"])
            acc_t = self.compute_accuracy(y_pred, batch_dict["target_y"])
            ho_running_acc += acc_t * len(batch_dict["x_seq"])

            self.train_state["heldout_preds"] += y_pred.tolist()
            self.train_state["heldout_true"] += batch_dict["target_y"].tolist()
            self.train_state["heldout_true_preds_id_tuple"] += list(
                zip(
                    batch_dict["krypht"],
                    y_pred.tolist(),
                    batch_dict["target_y"].tolist(),
                )
            )

        self.train_state["heldout_loss"] = ho_running_loss / len(self.dataset)
        self.train_state["heldout_acc"] = ho_running_acc / len(self.dataset)

        if self.writer is not None:
            self.writer.add_histogram(
                "ho y_preds", np.array(self.train_state["heldout_preds"]), 0
            )
            self.writer.add_histogram(
                "ho y_true", np.array(self.train_state["heldout_true"]), 0
            )

    def test(self):
        # TODO: switch running loss calculations
        self.dataset.set_split("test")
        batch_generator = self.dataset.generate_batches(
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.args.shuffle,
            device=self.args.device,
        )
        self.model.load_state_dict(torch.load(self.train_state["model_filename"]))
        self.model.eval()

        test_running_loss = RunningMean()
        test_running_acc = RunningMean()

        for batch_dict in batch_generator:
            # compute the output (forward pass)
            model_addln_args = self._get_model_addln_args(batch_dict)

            y_pred = self.model(
                batch_dict["x_seq"].to(self.args.device),
                batch_dict["seq_length"].to(self.args.device),
                **model_addln_args,
            )

            # compute the loss
            loss = self.loss_func(
                y_pred, batch_dict["target_y"].to(self.args.device).float()
            )

            test_running_loss.update(loss.item(), self.args.batch_size)
            test_running_acc.update(
                self.compute_accuracy(y_pred, batch_dict["target_y"]), args.batch_size
            )
            self.train_state["test_preds"] += y_pred.tolist()
            self.train_state["test_true"] += batch_dict["target_y"].tolist()

        self.train_state["test_loss"] = test_running_loss.value
        self.train_state["test_acc"] = test_running_acc.value

    def plot_performance(self, filename):
        # Figure size
        fig = plt.figure(figsize=(15, 5))
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(self.train_state["train_loss"], label="train")
        plt.plot(self.train_state["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend(loc="upper right")
        plt.rcParams.update({"font.size": 10})

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("R2 Score")
        plt.plot(self.train_state["train_acc"], label="train")
        plt.plot(self.train_state["val_acc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("R2")
        plt.legend(loc="lower right")

        fig.tight_layout(pad=1)
        fig.canvas.draw()

        # Save figure
        plt.savefig(os.path.join(self.args.save_dir, filename))

    def plot_residuals(self, filename):
        ytrue = self.train_state["heldout_true"]
        ypred = self.train_state["heldout_preds"]
        x = range(len(ytrue))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(
            ytrue,
            ypred,
            s=10,
            c="r",
            alpha=0.3,
            marker="o",
            label="predicted",
            cmap="viridis",
        )
        plt.xlabel("ytrue")
        plt.ylabel("ypred")
        plt.title("Residuals for predicted value")
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.args.save_dir, filename))
