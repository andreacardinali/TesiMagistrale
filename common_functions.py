from ctypes.wintypes import BOOL
import os
from pathlib import Path
import pickle
import string
from wsgiref import validate
from xml.etree.ElementInclude import include
import torch
import torchvision
import numpy as np
import time
import plotly.graph_objects as go
import re
from torch import nn
from torch.optim import Adam
from io import BytesIO

def disk_icon():
    import base64
    # This is the base64 encoded content of a 16x16 PNG floppy disk icon for save buttons
    base64_string = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAUVBMVEVHcEzs8PHM0tXc4OMlbp4rfbU0mNs0mNsrf7eavteu0ulxteIsiMTm6uzh5eedyudgq90ylNbA1uTV4+owkM/p7vBur9qqyt7B2+vX5u6tz+V8u4LNAAAACnRSTlMA////f5///Z/9cXypJgAAAF9JREFUGNONz8kOgCAMRVGLWqBVBmf9/w81FtCVCXd5Fi95TVOR0QwA8z5xl0DjA9eyMSZBAUJmxAxhBTjpCC/EQYovfCXwkPMZyCnJUYGxlcY/0FQ2SAv0VuWsqbl6A07jA+gGmk9/AAAAAElFTkSuQmCC"
    return BytesIO(base64.b64decode(base64_string))

def is_not_blank(s : str):
    return bool(s and not s.isspace())

def validate_filename(s):
    badchars = re.compile(r"[^A-Za-z0-9_. ]+|^\.|\.$|^ | $|^$")
    badnames = re.compile(r"(aux|com[1-9]|con|lpt[1-9]|prn)(\.|$)")
    name = badchars.sub("_", s)
    if badnames.match(name):
        name = "_" + name
    return name

def validate_filepath(filepath, mustexist : bool=True):
    if is_not_blank(str(filepath)):
        pathchars = re.compile(r'[\\/]')
        if pathchars.match(str(filepath)) and Path(filepath).is_file():
                return filepath
        # Check if it's a filepath or a filename
        elif pathchars.match(str(filepath)) is None:
            # input looks to be a filename only. Adding working dir and validate the path
            joined_path = Path(os.path.join(str(os.getcwd()), filepath))
            if joined_path.is_file():
                return joined_path
            else:
                if mustexist is True:
                    raise ValueError(f"Supplied path does not exist: {filepath}")
                else:
                    return joined_path
        else:
             raise ValueError(f"Supplied path is not valid: {filepath}")
    else:
        raise ValueError(f"Blank path was supplied.")

def save_model(model, model_name="model"):
    model_filename = validate_filename(model_name) + ".pth"
    model_filepath = validate_filepath(model_filename, False)
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_filepath) # Save
    print("Saved model to " + str(model_filepath))

def load_model(model_filepath, device="cpu"):
    model_filepath = validate_filepath(model_filepath)
    model = torch.jit.load(model_filepath, map_location=device)
    print("Loaded model from " + str(model_filepath))
    return model.eval()

def save_model_stats(model_name="model", train_loss=None, val_loss=None, accuracy=None, elapsedtime=None, epoch=None):
    stats_filename = validate_filename(model_name) + ".pkl"
    stats_filepath = validate_filepath(stats_filename, False)
    with open(stats_filepath, "wb") as f:
        pickle.dump([train_loss, val_loss, accuracy, elapsedtime, epoch, model_name], f)

def load_model_stats(stats_filename="model.pkl"):
    train_loss = None
    val_loss = None
    accuracy = None
    elapsedtime = None
    epoch = None
    model_name = None
    stats_filepath = validate_filepath(stats_filename)
    if Path(stats_filepath).is_file():
        with open(stats_filepath, "rb") as f:
            train_loss, val_loss, accuracy, elapsedtime, epoch, model_name = pickle.load(f)
    return train_loss, val_loss, accuracy, elapsedtime, epoch, model_name

# Training and Validation Function 
def train_model(model, num_epochs : int, train_dataloader, validation_dataloader, device="cpu", model_name : str="Model", prevent_overfit : bool=True, save_stats : bool=True):
    # Variables initializing
    overfit_tolerance = 10
    best_accuracy = 0.0
    winning_epoch = 0
    hystorical_train_loss = []
    hystorical_val_loss = []
    hystorical_accuracy = []
    hystorical_elapsedtime = []

    # Moving the model to the proper device
    model.to(device)
  
    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print(f"{model_name}: training started...") 
    for epoch in range(1, num_epochs + 1): 
        running_train_loss = 0.0 
        running_accuracy = 0.0 
        running_val_loss = 0.0 
        total = 0
        batch_size = train_dataloader.batch_size
        starttime = time.perf_counter()

        # Training Loop against train_dataloader
        # Switching to train mode
        model.train()
        for i, data in enumerate(train_dataloader): 
            # Get the images as input and real classes as outputs; data is a list of [inputs, outputs]
            inputs, outputs = data
            inputs, outputs = inputs.to(device), outputs.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Predicted output from the model 
            predicted_outputs = model(inputs)
            # Calculate loss for the predicted output
            train_loss = loss_fn(predicted_outputs, outputs)
            # Calculate the gradients and backpropagate the loss
            train_loss.backward()
            # Adjust optimizer parameters based on the calculated gradients
            optimizer.step()
            # Track the loss value
            running_train_loss += train_loss.item()
            # Print progress statistics every 500 images
            if i % int(500 / batch_size) == int(500 / batch_size) - 1:
                print(f"[TRAINING] {model_name}: {epoch}/{num_epochs} epochs, {(i + 1) * batch_size:5d}/{len(train_dataloader.dataset)} images - loss: {running_train_loss / (i + 1):.4f}", end = "\r")
        # Calculate final training loss value 
        train_loss_value = running_train_loss / len(train_dataloader)
        print("", end = "\n")
        
        # Validation Loop against validation_dataloader
        with torch.no_grad():
            # Switching to evaluation mode
            model.eval()
            for i, data in enumerate(validation_dataloader):
                # Get the images as input and real classes as outputs; data is a list of [inputs, outputs]
                inputs, outputs = data
                inputs, outputs = inputs.to(device), outputs.to(device)
                # Predicted output from the model
                predicted_outputs = model(inputs)
                # Calculate loss for the predicted output
                val_loss = loss_fn(predicted_outputs, outputs)
                # The label having the highest probability is the final prediction 
                _, predicted = torch.max(predicted_outputs, 1)
                # Track the loss value
                running_val_loss += val_loss.item()
                # Increase total counter
                total += outputs.size(0)
                # Track the running accuracy
                running_accuracy += (predicted == outputs).sum().item() 
                # Print progress statistics every 500 images
                if i % int(500 / batch_size) == int(500 / batch_size) - 1:
                    print(f"[VALIDATION] {model_name}: {epoch}/{num_epochs} epochs, {(i + 1) * batch_size:5d}/{len(validation_dataloader.dataset)} images - loss: {running_val_loss / (i + 1):.4f} - accuracy: {100 * running_accuracy / total:.2f} %", end = "\r")

        # Calculate final validation loss value
        val_loss_value = running_val_loss/len(validation_dataloader)
        print("", end = "\n")
                
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of correct predictions done
        accuracy = (100 * running_accuracy / total)     

        elapsedtime = time.perf_counter() - starttime

        # Print the epoch statistics
        print(f"[EPOCH {epoch} COMPLETE] Time: {elapsedtime:.2f} seconds. LOSSES: Training: {train_loss_value:.4f} - Validation: {val_loss_value:.4f} - ACCURACY: {accuracy:.2f} %")
        # Track the epoch statistics for optional save
        hystorical_train_loss.append(train_loss_value)
        hystorical_val_loss.append(val_loss_value)
        hystorical_accuracy.append(accuracy)
        hystorical_elapsedtime.append(elapsedtime)
        
        # Save the model if the accuracy is the best
        if epoch == 1:
            best_val_loss = val_loss_value

        val_loss_trend = (val_loss_value - best_val_loss) / best_val_loss * 100
        if val_loss_trend < -overfit_tolerance or epoch == 1:
            print(f"Model is not yet overfitting. Validation loss trend: {val_loss_trend:+.2f} %", end = "")
            best_val_loss = val_loss_value
        # The fit range is good....
        elif -overfit_tolerance <= val_loss_trend <= overfit_tolerance and epoch > 1:
            print(f"Model is in the best fitting range! Validation loss trend: {val_loss_trend:+.2f} % ", end = "")
            if accuracy > best_accuracy:
                print("")
                print("BEST ACCURACY SO FAR! Saving model...")
                save_model(model, model_name)
                winning_epoch = epoch
                best_accuracy = accuracy
        else:
            # The model is out of tolerance and is going to overfit
            print(f"Model is overfitting! Validation loss trend: {val_loss_trend:+.2f} %", end = "")
            if prevent_overfit is True:
                print("")
                print("Stopping training...")
                break
        print("")
    # Save statistics to file
    if save_stats is True:
        save_model_stats(model_name, hystorical_train_loss, hystorical_val_loss, hystorical_accuracy, hystorical_elapsedtime, winning_epoch)
    return hystorical_train_loss, hystorical_val_loss, hystorical_accuracy, hystorical_elapsedtime, winning_epoch

# Function to test the model and which classes are easier to predict  
def test_model(model, dataloader, device="cpu", print_progress : bool=True, class_stats : bool=True, attack=None): 
    # Moving the model to the proper device
    model.to(device)

    # Switching to evaluation mode
    model.eval() 

    # Variables initializing
    running_accuracy = 0 
    total = 0 
    batch_size = dataloader.batch_size
    starttime = time.perf_counter()

    # Class captions for labels
    labels = dataloader.dataset.class_to_idx
    # List for class captions
    label_list = list(labels.keys())
    # How many labels we have
    num_labels = len(labels)
    # List to calculate correct labels 
    labels_correct = list(0 for i in range(num_labels))
    # List to keep the total number of labels per type
    labels_total = list(0 for i in range(num_labels))
    for i, data in enumerate(dataloader):
        # Get the images as input and real classes as outputs; data is a list of [inputs, outputs]
        inputs, outputs = data
        inputs, outputs = inputs.to(device), outputs.to(device)
        outputs = outputs.to(torch.long)
        # Predicted output from the model, if attack is populated, use the attack function first in order to generate adversarial inputs
        if attack is None:
            predicted_outputs = model(inputs)
        else:
            adv_inputs = attack(inputs, outputs)
            predicted_outputs = model(adv_inputs)
        # The label having the highest probability is the final prediction
        _, predicted = torch.max(predicted_outputs, 1)
        # Increase total counter
        total += outputs.size(0)
        # Track the running accuracy
        running_accuracy += (predicted == outputs).sum().item()
        label_correct_running = (predicted == outputs).squeeze()
        # Print progress statistics every 500 images
        if i % int(500 / batch_size) == int(500 / batch_size) -1 and print_progress is True:
            print(f"[TESTING] Accuracy of the model based on the a set of {total} inputs is: {100 * running_accuracy / total:.2f} %", end="\r")
        
        # Calculates the accuracy all over the classes
        if class_stats is True:
            for idx, label_check in enumerate(label_correct_running):
                label = outputs[idx]
                labels_total[label] += 1
                if label_check:
                    labels_correct[label] += 1
    
    elapsedtime = time.perf_counter() - starttime
    print(f"[TESTING COMPLETE] Time: {elapsedtime:.2f} seconds. Accuracy of the model based on a set of", len(dataloader.dataset) ,"inputs is: %.2f %%" % (100 * running_accuracy / total))

    # Prints the class prediction statistics
    if class_stats is True:
        for i in range(num_labels):
            print(f"Accuracy to predict {label_list[i]} : {labels_correct[i]} out of {labels_total[i]} - {100 * labels_correct[i] / labels_total[i]:.2f}%")

def load_dataset(train: bool = True, normalize: bool = True):
    data_path = str(os.getcwd())
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True)
    tr_tensor = torchvision.transforms.ToTensor()
    if normalize is True:
        # Calculate mean and std for normalization
        mean, std = calculate_mean_std(dataset)
        tr_normalize = torchvision.transforms.Normalize(mean.tolist(), std.tolist())
        tr_unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        transform = torchvision.transforms.Compose([tr_tensor, tr_normalize])
    else:
        transform = tr_tensor

    dataset.transform = transform
    return dataset

def create_dataloader(train: bool = True, train_split: bool = True, shuffle: bool = False, batch_size: int = 50, normalize: bool = True, dataset = None):
    if dataset is None:
        dataset = load_dataset(train = train, normalize = normalize)

    if train is True and train_split is True:
        # Random splits the CIFAR10 train dataset in two subsets for training (60%) and validation (40%)
        train_subset_size = int(len(dataset)*0.6)
        validate_subset_size = len(dataset) - train_subset_size
        train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_subset_size, validate_subset_size])

        # Define the 2 dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle,
            collate_fn=None
        )

        validate_loader = torch.utils.data.DataLoader(
            validate_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle,
            collate_fn=None
        )
        return train_loader, validate_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle,
            collate_fn=None
        )
        return loader

cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
def get_cifar10_label_from_id(idx : int):
    return str(cifar10_labels[idx])

def get_cifar10_id_from_label(classname : str):
    return cifar10_labels.index(classname)

def path_is_class_label(path):
    directory_basename = os.path.basename(path)
    if directory_basename in cifar10_labels:
        print("Selected path has a CIFAR-10 class hierarchy")
        return directory_basename
    else:
        return None

def calculate_mean_std(tensor:torch.tensor):
    mean = torch.zeros([1, 3], dtype=torch.float32)
    std = torch.zeros([1, 3], dtype=torch.float32)
    if len(tensor.data.shape) == 3:
        mean = torch.tensor(tensor.data.mean(axis=(1,2)), dtype=torch.float32)
        std = torch.tensor(tensor.data.std(axis=(1,2)), dtype=torch.float32)
    elif len(tensor.data.shape) == 4:
        if tensor.data.shape[0] > 0:
            mean = torch.tensor(tensor.data.mean(axis=(0,1,2))/255, dtype=torch.float32)
            std = torch.tensor(tensor.data.std(axis=(0,1,2))/255, dtype=torch.float32)
        else:
            # tensor is a dataset
            tr_tensor = torchvision.transforms.ToTensor()
            tensor.transform = tr_tensor
            dataloader = torch.utils.data.DataLoader(tensor, batch_size=1, num_workers=1, shuffle=False, collate_fn=None)
            channels_sum, channels_squared_sum, num_batches = 0, 0, 0
            for data, _ in dataloader:
                # Mean over batch, height and width, but not over the channels
                channels_sum += torch.mean(data, dim=[0,2,3])
                channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
                num_batches += 1

            mean = channels_sum / num_batches

            # std = sqrt(E[X^2] - (E[X])^2)
            std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    else:
        raise ValueError("Tensor shape is not correct")

    return mean, std

def plot_training_results(train_loss, val_loss, accuracy, elapsedtime, epoch, model_name):
    total_x_ticks = len(train_loss)
    x = list(range(1, total_x_ticks + 1))

    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x = x,
            y = val_loss,
            name = "Validation Loss",
            yaxis = "y2",
            marker = dict(
                color = "blue",
                size = 8
            ),
            line = dict(
                color = "blue",
                width = 2
                ),
            mode = "lines+markers"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x,
            y = train_loss,
            name = "Train Loss",
            yaxis = "y2",
            marker = dict(
                color = "red",
                size = 8
            ),
            line = dict(
                color = "red",
                width = 2
                ),
            mode = "lines+markers"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = x,
            y = accuracy,
            name = "Accuracy",
            yaxis = "y1",
            marker = dict(
                color = "green",
                size = 8
            ),
            line = dict(
                color = "green",
                width = 2
                ),
            mode = "lines+markers"
            )
    )

    # Create axis objects
    fig.update_layout(
        title = dict(
            text = f"{model_name} - training curves",
            font = dict(size=50),
            yref = "paper"
        ),
        xaxis = dict(
            domain = [0, 1],
            tickmode = "linear",
            tick0 = 1,
            dtick = 1,
            title_text = "epoch",
            automargin = True
        ),
        yaxis = dict(
            title = "Accuracy",
            titlefont = dict(color="#0000ff"),
            tickfont = dict(color="#0000ff"),
            tickformat = ".2f",
            range = [0, 100],
            automargin = True
        ),
     
        # pass the y-axis 2 title, titlefont, color and
        # tickfont as a dictionary and store it an
        # variable yaxis 2
        yaxis2 = dict(
            title = "Loss",
            titlefont = dict(color="#FF0000"),
            tickfont = dict(color="#FF0000"),
            tickformat = ".4f",
            anchor = "x",
            overlaying = "y",
            side = "right"
        ),
        hovermode = "x unified"
    )

    # Best epoch zone
    fig.add_vline(x=epoch, line_width=5, line_dash="dash", line_color="orange", annotation_text="Best epoch", annotation_textangle=90)

    tol = min(val_loss) * 1.10
    x0 = list(filter(lambda i: i < tol, val_loss))[0]
    x1 = list(filter(lambda i: i < tol, val_loss))[-1]
    fig.add_vrect(
        x0 = x[val_loss.index(x0)],
        x1 = x[val_loss.index(x1)], 
        annotation_text = "Epochs within validation loss range",
        annotation_position = "bottom left",
        fillcolor = "green",
        opacity = 0.10,
        line_width = 0
    )

    fig.add_shape(
        type = "rect",
        yref = "y2",
        xref = "paper",
        y0 = min(val_loss),
        y1 = tol,
        x0 = 0,
        x1 = 1,
        fillcolor = "green",
        opacity = 0.10,
        line_width = 0
    )
    fig.add_annotation(
        yref = "y2",
        xref = "paper",
        y = min(val_loss),
        yshift = -10,
        x = 1,
        text = "Best validation loss range",
        showarrow = False
    )

    # Annotations
    fig.add_annotation(
        x = x[np.argmax(accuracy)],
        y = max(accuracy),
        text=f"Max Accuracy<br>epoch {x[np.argmax(accuracy)]}: {max(accuracy):.2f}",
        showarrow = True,
        font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
        align = "center",
        arrowhead = 2,
        arrowsize = 1,
        arrowwidth = 2,
        arrowcolor = "#636363",
        ax = 30,
        ay = -60,
        axref = "pixel",
        ayref = "pixel",
        bordercolor = "#c7c7c7",
        borderwidth = 2,
        borderpad = 4,
        bgcolor = "#ff7f0e",
        opacity = 0.6,
        xref = "x",
        yref = "y1"
    )

    fig.add_annotation(
        x = x[np.argmin(train_loss)],
        y = min(train_loss),
        text = f"Min Train loss<br>epoch {x[np.argmin(train_loss)]}: {min(train_loss):.4f}",
        showarrow = True,
        font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
        align = "center",
        arrowhead = 2,
        arrowsize = 1,
        arrowwidth = 2,
        arrowcolor = "#636363",
        ax = -30,
        ay = 60,
        axref = "pixel",
        ayref = "pixel",
        bordercolor = "#c7c7c7",
        borderwidth = 2,
        borderpad = 4,
        bgcolor = "#ff7f0e",
        opacity = 0.6,
        xref = "x",
        yref = "y2"
    )

    fig.add_annotation(
        x = x[np.argmin(val_loss)],
        y = min(val_loss),
        text = f"Min Validation loss<br>epoch {x[np.argmin(val_loss)]}: {min(val_loss):.4f}",
        showarrow = True,
        font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
        align = "center",
        arrowhead = 2,
        arrowsize = 1,
        arrowwidth = 2,
        arrowcolor = "#636363",
        ax = -30,
        ay = 60,
        axref = "pixel",
        ayref = "pixel",
        bordercolor = "#c7c7c7",
        borderwidth = 2,
        borderpad = 4,
        bgcolor = "#ff7f0e",
        opacity = 0.6,
        xref = "x",
        yref = "y2"
    )

    # Annotations for best epoch
    if epoch != x[np.argmax(accuracy)]:
        fig.add_annotation(
            x = epoch,
            y = accuracy[epoch-1],
            text = f"Accuracy:<br>{accuracy[epoch-1]:.2f}",
            showarrow = True,
            font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
            align = "center",
            arrowhead = 2,
            arrowsize = 1,
            arrowwidth = 2,
            arrowcolor = "#636363",
            ax = 30,
            ay = -60,
            axref = "pixel",
            ayref = "pixel",
            bordercolor = "#c7c7c7",
            borderwidth = 2,
            borderpad = 4,
            bgcolor = "#ff7f0e",
            opacity = 0.6,
            clicktoshow = "onout",
            visible = False,
            yref = "y1"
        )

    if epoch != x[np.argmin(train_loss)]:
        fig.add_annotation(
            x = epoch,
            y = train_loss[epoch-1],
            text = f"Train loss:<br>{train_loss[epoch-1]:.4f}",
            showarrow = True,
            font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
            align = "center",
            arrowhead = 2,
            arrowsize = 1,
            arrowwidth = 2,
            arrowcolor = "#636363",
            ax = -30,
            ay = 60,
            axref = "pixel",
            ayref = "pixel",
            bordercolor = "#c7c7c7",
            borderwidth = 2,
            borderpad = 4,
            bgcolor = "#ff7f0e",
            opacity = 0.6,
            clicktoshow = "onout",
            visible = False,
            yref = "y2"
        )
    if epoch != x[np.argmin(val_loss)]:
        fig.add_annotation(
            x = epoch,
            y = val_loss[epoch-1],
            text = f"Validation loss:<br>{val_loss[epoch-1]:.4f}",
            showarrow = True,
            font = dict(
                family = "Courier New, monospace",
                size = 16,
                color = "#ffffff"
                ),
            align = "center",
            arrowhead = 2,
            arrowsize = 1,
            arrowwidth = 2,
            arrowcolor = "#636363",
            ax = -30,
            ay = 60,
            bordercolor = "#c7c7c7",
            borderwidth = 2,
            borderpad = 4,
            bgcolor = "#ff7f0e",
            opacity = 0.6,
            clicktoshow = "onout",
            visible = False,
            yref = "y2"
        )
    
    fig.show()
