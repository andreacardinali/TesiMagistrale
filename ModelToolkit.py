from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_ImageTk
import numpy as np
import sys, threading
import os
import time
import torch
import torchattacks
import common_functions as C
from pathlib import Path
from fnmatch import fnmatch
from torchvision import transforms
from torchvision.utils import save_image as torch_save_image, make_grid as torch_make_grid
from tkinter import Tk, StringVar, BooleanVar, IntVar, Frame, Text, Button, Radiobutton, Checkbutton, OptionMenu, Label, Entry, Scrollbar, messagebox, filedialog

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see("end")

    def flush(self):
        pass

class App(Tk):

    def classify_image(self):
        if hasattr(self, "t_source_image") and torch.is_tensor(self.t_source_image):
            self.recognize_image_single(image=self.t_source_image, source=True)
            if hasattr(self, "t_adv_image") and torch.is_tensor(self.t_adv_image):
                self.recognize_image_single(image=self.t_adv_image, source=False)
           
    def recognize_image(self, image):
        if image is not None and self.model is not None and self.device is not None:
            if torch.is_tensor(image) or isinstance(image, PIL_Image.Image):
                img_tensor = self.preprocess_image(image)
            elif Path(image).is_file():
                img_tensor = self.load_imagefile(image, self.device)
            else:
                pass

            if img_tensor.dim() == 3:
                img_tensor= img_tensor.unsqueeze(0)

            with torch.no_grad():
                output = self.model(img_tensor)
                _, predict = torch.max(output, 1)
                predictions = torch.nn.functional.softmax(output[0], dim=0)
                return predict.item(), predictions[predict].item()
        else:
            raise ValueError("Image, model or device are not set.")
                
    def recognize_image_single(self, image, source=True):
        if image is not None and self.model is not None:
            predict, predictions = self.recognize_image(image)
  
            prediction_label = C.get_cifar10_label_from_id(predict)
            prediction_text = f"{prediction_label} {(predictions * 100):.2f}%"
            if source:
                self.sourceImage_prediction.set(str(prediction_text))
            else:
                self.adversarialImage_prediction.set(str(prediction_text))

            self.set_classification()
            if C.is_not_blank(self.source_classification.get()):
                if self.source_classification.get() == prediction_label:
                    if source:
                        self.sourceImage_predictionlabel.config(fg="green")
                    else:
                        self.adversarialImage_predictionlabel.config(fg="green")
                    return True
                else:
                    if source:
                        self.sourceImage_predictionlabel.config(fg="red")
                    else:
                        self.adversarialImage_predictionlabel.config(fg="red")
                    return False

    def recognize_image_dir(self):
        if self.dir is not None and len(self.files) > 0 and self.class_from_dir is not None and self.model is not None:
            total = len(self.files)
            i = 0
            goodmatches = 0
            self.set_classification()
            if C.is_not_blank(self.class_from_dir) and self.class_from_dir == self.source_classification.get():
                print(f"Found {total} images to scan. Correct class is {self.class_from_dir}")
            elif C.is_not_blank(self.source_classification.get()):
                print(f"Found {total} images to scan. Overriden class is {self.source_classification.get()}")
            else:
                raise ValueError("Unable to determine class. Please select manually.")
            for image in self.files:
                i += 1
                pct = i/total * 100
                if pct % 5 == 0:
                    self.update_idletasks()
                    if pct % 20 == 0:
                        print("!", end="\r")
                    else:
                        print("o", end="\r")
                
                predict, _ = self.recognize_image(image)
  
                label = C.get_cifar10_label_from_id(predict)
                
                if C.is_not_blank(self.source_classification.get()) and self.source_classification.get() == label:
                    goodmatches += 1
            print("", end="\r\n")
            result = goodmatches / total
            result_string = f"Completed. Found {goodmatches} correct matches out of {total} images ({(result * 100):.2f}%)"
            print(result_string)

    def recognize_image_seek(self, operation: str = "classify"):
        proceed = True
        if not operation in ("attack", "classify"):
            raise ValueError("Invalid operation specified!")

        if self.filemode.get() is True and self.dir is not None and len(self.files) > 0 and self.class_from_dir is not None and self.model is not None:
            total = len(self.files)
            print(f"Found {total} images to {operation}. Correct class is {self.class_from_dir}")
            if total >= 1000:
                proceed = messagebox.askyesno("Warning", f"Found {total} images to {operation}. This operation will iterate all the source images one by one and will take some time to complete.\n\nThe app will look unresponsive until finished. Do you still want to proceed?")
        elif self.filemode.get() is False and self.model is not None:
            total = len(self.dataset)
            print(f"Found {total} images to {operation}.")
            if total >= 1000:
                proceed = messagebox.askyesno("Warning", f"Found {total} images to {operation}. This operation will iterate all the source images one by one and will take some time to complete.\n\nThe app will look unresponsive until finished.\n\"{operation.capitalize()} dataset\" is much way faster, as it will process the dataset in batches.\n\nDo you still want to proceed?")
        else:
            raise ValueError("Either model/dataset/directory is missing or directory has not a CIFAR-10 hierarchy")
        
        old_img_index = self.img_index
        if proceed is True:
            self.set_classification()
            self.attack_on_the_fly.set(False)
            starttime = time.perf_counter()
            i = 0
            goodmatches = 0
            for i in range(0, total):
                self.seek(img_index=i, batch=True)

                if operation == "attack":
                    self.attack_image(image=self.t_source_image, batch=True)
                    predict, _ = self.recognize_image(self.t_adv_image)
                else:
                    predict, _ = self.recognize_image(self.t_source_image)
  
                label = C.get_cifar10_label_from_id(predict)

                if C.is_not_blank(self.source_classification.get()) and self.source_classification.get() == label:
                    goodmatches += 1
            
            print("", end="\r\n")
            result = goodmatches / total
            elapsedtime = time.perf_counter() - starttime
            result_string = f"Completed in {elapsedtime:.2f} seconds. Found {goodmatches} correct matches out of {total} images ({(result * 100):.2f}%)"
            print(result_string)
        else:
            result_string = f"{operation.capitalize()} all images was aborted."
            print(result_string)
        self.seek(old_img_index)

    def attack_image_seek(self):
        print("Attack type: " + str(self.attacktype.get()))
        self.recognize_image_seek(operation="attack")
                
    def attack_inspect(self):
        if hasattr(self, "attacktype") and self.attacktype.get() != "":
            import inspect
            self.attack_function = getattr(torchattacks, str(self.attacktype.get()))
            self.attack_function_args = inspect.getfullargspec(self.attack_function)
            self.attack_function_sig = inspect.signature(self.attack_function)

            self.attack_args_values = []
            self.attack_args_names = []

            for i in range(self.maxargs):
                if self.attack_args_menu[i] is not None:
                    self.attack_args_menu[i].grid_forget()
                if self.attack_args_labels[i] is not None:
                    self.attack_args_labels[i].grid_forget()

            i = 0
            for param in self.attack_function_sig.parameters.values():
                if param.default is not param.empty and i < self.maxargs:
                    self.attack_args_values.insert(i, StringVar())
                    self.attack_args_values[i].set(param.default)
                    self.attack_args_names.insert(i, StringVar())
                    self.attack_args_names[i].set(param.name)
                    self.attack_args_labels.insert(i, Label(self.attack_menu, textvariable=self.attack_args_names[i]))
                    self.attack_args_labels[i].grid(row=2, column=i, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
                    self.attack_args_menu.insert(i, Entry(self.attack_menu, textvariable=self.attack_args_values[i], width=10))
                    self.attack_args_menu[i].grid(row=3, column=i, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
                    self.attack_args_menu[i].bind("<Return>", lambda *e: self.attack_image())
                    i+=1
            self.targeted_attack_enable.set(True)
        self.refresh_gui()
        self.refresh_attack()
    
    def build_attack_function(self):
        kwargs={}
        i = 0
        for param in self.attack_function_sig.parameters.values():
            if param.default is not param.empty and i < self.maxargs:
                if self.attack_args_names[i] is not None:
                    argvalue = str(self.attack_args_values[i].get())
                    if C.is_not_blank(argvalue) is False:
                        argvalue = None
                    elif isinstance(param.default, int):
                        argvalue = int(self.attack_args_values[i].get())
                    elif isinstance(param.default, float):
                        argvalue = float(self.attack_args_values[i].get())
                    elif isinstance(param.default, bool):
                        argvalue = bool(self.attack_args_values[i].get())
                    kwargs[param.name]=argvalue
                i+=1

        attack = self.attack_function(self.model, **kwargs)

        # If images are normalized:
        if self.use_normalization.get() is True:
            attack.set_normalization_used(mean=self.mean, std=self.std)

        return attack
            
    def attack_image(self, image=None, batch=False):
        tr_unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        if image is None:
            image = self.t_source_image

        if not torch.is_tensor(image):
            raise ValueError("Image must be a tensor")

        self.t_delta_image = None
        self.t_adv_image = None
        self.t_adv_image_u = None
        self.attack = None
        if image.dim() == 3:
            image_4d = image.unsqueeze(0)
        elif image.dim() == 4:
            image_4d = image
        else:
            pass

        if batch is False:
            print("Attack type: " + str(self.attacktype.get()))
        
        attack = self.build_attack_function()
        
        # Define source label
        if C.is_not_blank(self.source_classification.get()):
            label = torch.tensor([C.get_cifar10_id_from_label(self.source_classification.get())])
        else:
            messagebox.showerror("Attack image", "Attack requires a source class. Please select manually")
            return

        if self.targeted_attack.get() is True:
            try:
                if not isinstance(self.target_classification.get(), str):
                    raise ValueError("Please select a target class")
                attack.set_mode_targeted_by_label(quiet=True) # do not show the message
                if batch is False:
                    print("Targeted label: " + self.target_classification.get() + " " + str(C.get_cifar10_id_from_label(self.target_classification.get())))
                label = torch.tensor([C.get_cifar10_id_from_label(self.target_classification.get())])
            except ValueError as e:
                print(e)
                self.targeted_attack.set(False)
                self.target_classification.set("")
                self.targeted_attack_enable.set(False)
                self.refresh_gui()
                return
        
        t_adv_image = attack(image_4d, label)
        t_delta_image = t_adv_image - image_4d
        # Squeezes back to 3 dim
        self.t_adv_image = t_adv_image.squeeze(0)
        self.t_delta_image = t_delta_image.squeeze(0)
        self.attack = attack

        if self.use_normalization.get() is True:
            self.t_adv_image_u = tr_unnormalize(self.t_adv_image)

        if batch is False:
            print(attack)
            self.refresh_classify()
            
    def open_file(self):
        load_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        filepath = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")])
        if filepath != "" and Path(filepath).is_file():
            directory = os.path.dirname(filepath)
            self.dir = os.path.normpath(directory)
            self.class_from_dir = C.path_is_class_label(self.dir)
            files = [os.path.normpath(f.path) for f in os.scandir(directory) if any(fnmatch(f, p) for p in load_extensions)]
            self.files = files
            img_filepath = os.path.normpath(filepath)
            self.img_index = self.files.index(img_filepath)
            self.max_index = len(files) - 1
            self.seek(img_index=self.img_index)

    def save_log(self):    
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="log.txt", filetypes=[("Text files", "*.txt")])
        if C.is_not_blank(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.log_text.get("1.0", "end"))

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
            
    def save_file(self, imagetype : str):
        if imagetype not in ("source", "source_u", "delta", "adversarial", "adversarial_u", "grid"):
            raise ValueError("imagetype is not valid")
        if self.filemode.get() is True:
            initialfile = str(C.validate_filename(Path(self.img_filename.get()).stem) + "_" + imagetype)
        else:
            initialfile = str(self.img_filename.get()) + "_" + imagetype

        filepath = filedialog.asksaveasfilename(defaultextension=".png", initialfile=initialfile, filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")])

        if C.is_not_blank(filepath):
            import matplotlib.pyplot as plt
            save_extensions = (".png", ".jpg", ".bmp")
            default_ext = ".png"
            filepath = os.path.normpath(filepath) if (os.path.splitext(filepath)[-1]).lower() in save_extensions else os.path.normpath(filepath + default_ext)

            if imagetype == "source":
                self.PIL_source_image.save(filepath)
            if imagetype == "source_u":
                self.save_imagefile(self.PIL_source_image_u, filepath)
            if imagetype == "delta":
                self.save_imagefile(self.PIL_delta_image, filepath)
            if imagetype == "adversarial":
                self.save_imagefile(self.PIL_adv_image, filepath)
            if imagetype == "adversarial_u":
                self.save_imagefile(self.PIL_adv_image_u, filepath)
            if imagetype == "grid":
                PIL_list = []
                title = ""
                if self.t_source_image_u is not None and torch.is_tensor(self.t_source_image_u):
                    PIL_list.append(self.PIL_source_image_u)
                if self.t_source_image is not None and torch.is_tensor(self.t_source_image):
                    PIL_list.append(self.PIL_source_image)
                if self.t_delta_image is not None and torch.is_tensor(self.t_delta_image):
                    PIL_list.append(self.PIL_delta_image)
                if self.t_adv_image is not None and torch.is_tensor(self.t_adv_image):
                    title = str(self.attack)
                    PIL_list.append(self.PIL_adv_image)
                if self.t_adv_image_u is not None and torch.is_tensor(self.t_adv_image_u):
                    PIL_list.append(self.PIL_adv_image_u)
                if len(PIL_list) > 0:
                    fig, ax = plt.subplots(1, len(PIL_list))
                    fig.suptitle(title, wrap=True)
                    fig.patch.set_facecolor("white")

                    for index, img in enumerate(PIL_list):
       
                        image = np.asarray(img)
                        ax[index].imshow(image, cmap=plt.cm.binary, aspect="equal")
                        ax[index].set_xticks([])
                        ax[index].set_yticks([])
                        ax[index].use_sticky_edges = False
                        ax[index].patch.set_facecolor("None")
                    
                    fig.savefig(filepath, facecolor="white", transparent=True)
            
    def open_dir(self):
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        directory = filedialog.askdirectory(mustexist=True)
        if C.is_not_blank(directory) and Path(directory).is_dir():
            files = [os.path.normpath(f.path) for f in os.scandir(directory) if any(fnmatch(f, p) for p in patterns)]
            if len(files) > 0:
                self.dir = os.path.normpath(directory)
                self.class_from_dir = C.path_is_class_label(self.dir)
                self.files = files
                self.img_index = 0
                self.max_index = len(files) - 1
                self.seek(img_index=self.img_index)
            else:
                messagebox.showerror("Open folder", "No valid image files found in this folder!")

    def open_model(self):
        model_filepath = os.path.normpath(filedialog.askopenfilename(filetypes=[("Model files", "*.pth")]))
        if C.is_not_blank(model_filepath) and Path(model_filepath).is_file():
            self.model = C.load_model(model_filepath, self.device)
            self.model_filepath.set(str(model_filepath))
            self.model_name = Path(model_filepath).stem
            
            # Check for training stats
            stats_filepath = os.path.splitext(str(model_filepath))[0] + '.pkl'
            if C.is_not_blank(stats_filepath) and Path(stats_filepath).is_file():
                self.stats_filepath.set(stats_filepath)
                print("Found training statistics: " + stats_filepath)

            self.refresh_classify()
            self.refresh_attack()

    def open_dataset(self, dataset_type:str = "train"):
        if dataset_type.lower() not in ("train", "test"):
            raise ValueError("Invalid dataset type supplied")
        if dataset_type.lower() == "train":
            train = True
        else:
            train = False

        self.dataset_type = dataset_type.lower()
        self.dataloader = C.create_dataloader(train=train, train_split=False, normalize=self.use_normalization.get(), batch_size=self.batch_size)
        self.dataset = self.dataloader.dataset
        self.mean, self.std = C.calculate_mean_std(self.dataset)
        self.img_index = 0
        self.max_index = len(self.dataset) - 1
        self.seek(img_index=self.img_index)
            
    def classify_dataset(self):
        print(f"Classifying {self.dataset_type} dataset.")
        C.test_model(model=self.model, dataloader=self.dataloader, device=self.device, print_progress=False) 

    def attack_dataset(self):
        print(f"Classifying {self.dataset_type} dataset against {str(self.attacktype.get())} attack:")
        attack = self.build_attack_function()
        print(attack)
        # Refresh GUI
        self.update_idletasks()
        C.test_model(model=self.model, dataloader=self.dataloader, device=self.device, print_progress=False, attack=attack)
    
    def show_training_curves(self):
        stats_filepath = C.validate_filepath(str(self.stats_filepath.get()))
        C.plot_training_results(*C.load_model_stats(stats_filepath))

    def unlockbuttons(self):
        # Disables all buttons and then enables only the needed ones
        self.bOpenTrainDataset.config(state="disabled")
        self.bOpenTestDataset.config(state="disabled")
        self.bOpenFile.config(state="disabled")
        self.bOpenDir.config(state="disabled")
        self.bPrev.config(state="disabled")
        self.index_check.config(state="disabled")
        self.bNext.config(state="disabled")
        self.bZoomDown.config(state="disabled")
        self.bZoomUp.config(state="disabled")
        self.bShowTrainingCurves.config(state="disabled")
        self.bClassify.config(state="disabled")
        self.bClassifyAll.config(state="disabled")
        self.bClassifyDataset.config(state="disabled")
        self.bAttack.config(state="disabled")
        self.bAttackAll.config(state="disabled")
        self.bAttackDataset.config(state="disabled")
        self.bCheckTargetedAttack.config(state="disabled")
        self.bCheckTargetedAttack.grid_remove()
        
        if self.filemode.get() is True:
            self.bOpenFile.config(state="normal")
            self.bOpenDir.config(state="normal")
            if len(self.files) > 0:
                self.bPrev.config(state="normal")
                self.index_check.config(state="normal")
                self.bNext.config(state="normal")
                self.bCheckOverride.grid()
                self.bCheckOverride.config(state="normal")
                if self.zoomlevel.get() < self.max_zoom:
                    self.bZoomUp.config(state="normal")
                if self.zoomlevel.get() > 1:
                    self.bZoomDown.config(state="normal")
                if self.model is not None:
                    self.bClassify.config(state="normal")
                    self.bClassifyAll.config(state="normal")
                    self.bCheckClassify.config(state="normal")
                    self.bCheckAttack.config(state="normal")
                    self.bAttack.config(state="normal")
                    self.bAttackAll.config(state="normal")
                    if self.targeted_attack_enable.get() is True:
                        self.bCheckTargetedAttack.grid()
                        self.bCheckTargetedAttack.config(state="normal")
                
        if self.filemode.get() is False:
            self.bOpenTrainDataset.config(state="normal")
            self.bOpenTestDataset.config(state="normal")
            if self.dataset is not None and len(self.dataset) > 0:
                self.bPrev.config(state="normal")
                self.index_check.config(state="normal")
                self.bNext.config(state="normal")
                self.bCheckOverride.grid()
                self.bCheckOverride.config(state="normal")
                if self.zoomlevel.get() < self.max_zoom:
                    self.bZoomUp.config(state="normal")
                if self.zoomlevel.get() > 1:
                    self.bZoomDown.config(state="normal")
                if self.model is not None:
                    self.bClassify.config(state="normal")
                    self.bClassifyAll.config(state="normal")
                    self.bClassifyDataset.config(state="normal")
                    self.bCheckClassify.config(state="normal")
                    self.bCheckAttack.config(state="normal")
                    self.bAttack.config(state="normal")
                    self.bAttackAll.config(state="normal")
                    self.bAttackDataset.config(state="normal")
                    if self.targeted_attack_enable.get() is True:
                        self.bCheckTargetedAttack.grid()
                        self.bCheckTargetedAttack.config(state="normal")
        
        if self.model is not None and C.is_not_blank(self.stats_filepath.get()) and Path(self.stats_filepath.get()).is_file():
            self.bShowTrainingCurves.config(state="normal")

        if self.override_classification.get() is False:
            self.override_class_menu.config(state="disabled")
            self.override_class_menu.grid_remove()

        if self.override_classification.get() is True:
            self.override_class_menu.config(state="normal")
            self.override_class_menu.grid()
            if not C.is_not_blank(self.source_classification_override.get()):
                self.source_classification_override.set(self.class_list[0])
        
        if self.targeted_attack.get() is False:
            self.targeted_class_menu.config(state="disabled")
            self.targeted_class_menu.grid_remove()

        if self.targeted_attack.get() is True:
            self.targeted_class_menu.config(state="normal")
            self.targeted_class_menu.grid()
            if not C.is_not_blank(self.target_classification.get()):
                self.target_classification.set(self.class_list[0])

        # hiding and disabling unnormalized GUI elements is no normalization is used
        if self.use_normalization.get() is False:
            self.sourceImage.grid(row=1, column=0, padx=5, pady=5)
            self.sourceImage_predictionlabel.grid(row=2, column=0, columnspan=1, padx=5, pady=5)
            self.bSaveSourceImage.grid(row=0, column=0, columnspan=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.bSaveSourceImageUnnormalized.grid(row=0, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.img_filenamelabel.grid(row=3, column=0, columnspan=1, padx=5, pady=5)
            self.sourceImageUnnormalized.grid_remove()
            self.adversarialImageUnnormalized.grid_remove()
        else:
            self.sourceImage.grid(row=1, column=1, padx=5, pady=5)
            self.sourceImage_predictionlabel.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
            self.bSaveSourceImage.grid(row=0, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.bSaveSourceImageUnnormalized.grid(row=0, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.img_filenamelabel.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
            self.sourceImageUnnormalized.grid()
            self.adversarialImageUnnormalized.grid()

        self.bSaveSourceImage.grid_remove()
        self.bSaveSourceImageUnnormalized.grid_remove()
        self.bSaveDeltaImage.grid_remove()
        self.bSaveAdversarialImage.grid_remove()
        self.bSaveAdversarialImageUnnormalized.grid_remove()

    def set_classification(self):
        self.source_classification.set("")
        if self.override_classification.get() is False:
            if C.is_not_blank(self.source_classification_fromsource.get()):
                self.source_classification.set(self.source_classification_fromsource.get())
        elif C.is_not_blank(self.source_classification_override.get()):
            self.source_classification.set(self.source_classification_override.get())

    def load_imagefile(self, filepath, device="cpu"):
        # Read a PIL image
        im = PIL_Image.open(filepath)
        # Preprocessing
        img_tensor = self.preprocess_image(im)
        img_tensor.to(device)
        return img_tensor

    def save_imagefile(self, image, filepath, normalize:bool=False):
        try:
            # Save the image
            if torch.is_tensor(image):
                torch_save_image(tensor=image, fp=filepath, normalize=normalize)
            elif isinstance(image, PIL_Image.Image):
                image.save(fp=filepath)    
            messagebox.showinfo("Image save", f"The image has been saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Image save", f"An error occured while saving the image:\n{e}")

    def save_grid(self):
        tr_PIL = transforms.ToPILImage()
        grid_list = []
        if self.t_source_image_u is not None and torch.is_tensor(self.t_source_image_u):
            grid_list.append(self.t_source_image_u.squeeze(0))
        if self.t_source_image is not None and torch.is_tensor(self.t_source_image):
            grid_list.append(self.t_source_image.squeeze(0))
        if self.t_delta_image is not None and torch.is_tensor(self.t_delta_image):
            grid_list.append(self.t_delta_image.squeeze(0))
        if self.t_adv_image is not None and torch.is_tensor(self.t_adv_image):
            grid_list.append(self.t_adv_image.squeeze(0))
        if self.t_adv_image_u is not None and torch.is_tensor(self.t_adv_image_u):
            grid_list.append(self.t_adv_image_u.squeeze(0))

        print(len(grid_list))

        if len(grid_list) > 0:
            t_grid = torch_make_grid(grid_list)
            print(t_grid.size())
            PIL_grid = tr_PIL(t_grid)
            PIL_grid.show()

    def preprocess_image(self, image) -> torch.tensor:
        tr_tensor = transforms.ToTensor()
        tr_resize = transforms.Resize(size=(32,32), antialias=None)

        if torch.is_tensor(image):
            if image.shape[-2] > 32 or image.shape[-1] > 32:
                image = tr_resize(image)
                print("Image size was greater than 32 pixel per side. Resized.")
        elif isinstance(image, PIL_Image.Image):
        # preprocessing and converting the image to tensor
            if image.mode != "RGB":
                image = image.convert("RGB")
                print("Image has been converted to RGB format.")
            if image.width > 32 or image.height > 32:
                image = tr_resize(image)
                print("Image has been resized to 32x32 resolution.")
            image = tr_tensor(image)
        else:
            raise ValueError("Image is either not a tensor or a PIL Image.")
        return image

    def tensor_to_PIL(self, image:torch.Tensor):
        # Using transforms.ToPILImage() creates weird colors for normalized images.
        # 
        # Taken from https://pytorch.org/vision/0.8/_modules/torchvision/utils.html#save_image
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = PIL_Image.fromarray(ndarr)
        return im

    def black_to_transparency(self, img):
        x = np.asarray(img.convert("RGBA")).copy()
        x[:, :, 3] = (255 * (x[:, :, :3] != 0).any(axis=2)).astype(np.uint8)
        return PIL_Image.fromarray(x)

    def white_to_transparency(self, img):
        x = np.asarray(img.convert("RGBA")).copy()
        x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
        return PIL_Image.fromarray(x)

    def black_to_transparency_tensor(self, img):
        b = img.cpu.detach()
        alpha = torch.Tensor(1, b.shape[1], b.shape[2])
        x = torch.cat(b, alpha, 1)
        x[:, :, 3] = (255 * (x[:, :, :3] != 0).any(axis=2)).type(torch.uint8)
        return x

    def seek(self, img_index:int=0, batch=False):
        self.t_delta_image = None
        self.t_adv_image = None
        self.t_adv_image_u = None
        if self.filemode.get() is True:
            img_filename = self.files[img_index]
            self.img_filename.set(str(img_filename))
            t_source_image_3d = self.load_imagefile(str(img_filename), self.device)
            # Preprocessing
            self.mean, self.std = C.calculate_mean_std(t_source_image_3d)
            if self.use_normalization.get() is True:
                tr_normalize = transforms.Normalize(self.mean.tolist(), self.std.tolist())
                self.t_source_image = tr_normalize(t_source_image_3d).to(self.device)
                self.t_source_image_u = t_source_image_3d.to(self.device)
                self.PIL_source_image = self.tensor_to_PIL(self.t_source_image)
                self.PIL_source_image_u = self.tensor_to_PIL(self.t_source_image_u)
            else:
                self.t_source_image = t_source_image_3d.to(self.device)
                self.t_source_image_u = None
                self.PIL_source_image = self.tensor_to_PIL(self.t_source_image)
                self.PIL_source_image_u = None
        else:
            # Human readable index
            img_filename = "Dataset Image #" + str(img_index + 1)
            self.img_filename.set(str(img_filename))
            dataset_item = self.dataset[img_index]
            dataset_image = dataset_item[0]
            dataset_target = dataset_item[1]
            if batch is True:
                self.source_classification.set(C.get_cifar10_label_from_id(dataset_target))
            else:
                self.source_classification_fromsource.set(C.get_cifar10_label_from_id(dataset_target))
            # Dataloader loads images with or without normalization already applied
            t_source_image_3d = dataset_image
            self.PIL_source_image = self.tensor_to_PIL(t_source_image_3d)
            self.t_source_image = t_source_image_3d.to(self.device)
            if self.use_normalization.get() is True:
                tr_unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
                self.t_source_image_u = tr_unnormalize(t_source_image_3d).to(self.device)
                self.PIL_source_image_u = self.tensor_to_PIL(self.t_source_image_u)
            else:
                # Dataloader already loads images with no normalization
                self.t_source_image_u = None
                self.PIL_source_image_u = None
        
        if batch is False:
            self.img_filename_display.set(str(img_filename))
            self.refresh_classify()
            self.refresh_attack()

    def seek_prev(self):
        self.img_index = self.img_index-1
        if self.img_index < 0:
            self.img_index = self.max_index
        self.seek(img_index=self.img_index)
        
    def seek_next(self):
        self.img_index = self.img_index + 1
        if self.img_index > self.max_index:
            self.img_index = 0
        self.seek(img_index=self.img_index)

    def seek_to(self):
        old_img_index = self.img_index
        
        try:
            self.img_index_human.set(self.index_check.get())
            self.img_index = int(self.img_index_human.get() - 1)
            self.seek(img_index=self.img_index)
        except Exception:
            self.img_index = old_img_index
            self.img_index_human.set(old_img_index + 1)
            self.seek(img_index=self.img_index)
        
    def zoomup(self):
        zoomlevel = self.zoomlevel.get()
        zoomlevel = zoomlevel + 1
        if zoomlevel > self.max_zoom:
            self.zoomlevel.set(8)
        else:
            self.zoomlevel.set(zoomlevel)

    def zoomdown(self):
        zoomlevel = self.zoomlevel.get()
        zoomlevel = zoomlevel - 1
        if zoomlevel < 1:
            self.zoomlevel.set(1)
        else:
            self.zoomlevel.set(zoomlevel)

    def zoomer(self):
        zoomlevel = self.zoomlevel.get()

        if hasattr(self, "PIL_source_image") and isinstance(self.PIL_source_image, PIL_Image.Image):
            if self.PIL_source_image.mode == "1": # bitmap image
                Tk_source_image = PIL_ImageTk.BitmapImage(self.PIL_source_image, foreground="white")
            else:              # photo image
                Tk_source_image = PIL_ImageTk.PhotoImage(self.PIL_source_image)
            self.Tk_source_image = Tk_source_image._PhotoImage__photo.zoom(zoomlevel, zoomlevel)
            self.sourceImage.config(image=self.Tk_source_image, width=self.Tk_source_image.width(), height=self.Tk_source_image.height())
            self.bSaveSourceImage.grid()
            self.bSaveAll.grid()

            if self.use_normalization.get() is True:
                if self.PIL_source_image_u.mode == "1": # bitmap image
                    Tk_source_image_u = PIL_ImageTk.BitmapImage(self.PIL_source_image_u, foreground="white")
                else:              # photo image
                    Tk_source_image_u = PIL_ImageTk.PhotoImage(self.PIL_source_image_u)
                self.Tk_source_image_u = Tk_source_image_u._PhotoImage__photo.zoom(zoomlevel, zoomlevel)
                self.sourceImageUnnormalized.config(image=self.Tk_source_image_u, width=self.Tk_source_image_u.width(), height=self.Tk_source_image_u.height())
                self.bSaveSourceImageUnnormalized.grid()
        
        if hasattr(self, "t_delta_image") and torch.is_tensor(self.t_delta_image):
            t_delta_image_3d = self.t_delta_image
            self.PIL_delta_image = self.black_to_transparency(self.tensor_to_PIL(t_delta_image_3d))
            Tk_delta_image = PIL_ImageTk.PhotoImage(self.PIL_delta_image)
            self.Tk_delta_image = Tk_delta_image._PhotoImage__photo.zoom(zoomlevel, zoomlevel)
            self.deltaImage.config(image=self.Tk_delta_image, width=self.Tk_delta_image.width(), height=self.Tk_delta_image.height())
            self.bSaveDeltaImage.grid()
        else:
            self.deltaImage.config(image="", width=0, height=0)
            self.bSaveDeltaImage.grid_remove()
            self.bSaveAdversarialImageUnnormalized.grid_remove()

        if hasattr(self, "t_adv_image") and torch.is_tensor(self.t_adv_image):
            t_adv_image_3d = self.t_adv_image
            self.PIL_adv_image = self.tensor_to_PIL(t_adv_image_3d)
            Tk_adv_image = PIL_ImageTk.PhotoImage(self.PIL_adv_image)
            self.Tk_adv_image = Tk_adv_image._PhotoImage__photo.zoom(zoomlevel, zoomlevel)
            self.adversarialImage.config(image=self.Tk_adv_image, width=self.Tk_adv_image.width(), height=self.Tk_adv_image.height())
            self.bSaveAdversarialImage.grid()
            
            if self.use_normalization.get() is True:
                t_adv_image_u_3d = self.t_adv_image_u
                PIL_adv_image_u = self.tensor_to_PIL(t_adv_image_u_3d)
                Tk_adv_image_u = PIL_ImageTk.PhotoImage(PIL_adv_image_u)
                self.Tk_adv_image_u = Tk_adv_image_u._PhotoImage__photo.zoom(zoomlevel, zoomlevel)
                self.adversarialImageUnnormalized.config(image=self.Tk_adv_image_u, width=self.Tk_adv_image_u.width(), height=self.Tk_adv_image_u.height())
                self.bSaveAdversarialImageUnnormalized.grid()
        else:
            self.adversarialImage.config(image="", width=0, height=0)
            self.adversarialImage_prediction.set("")
            self.adversarialImageUnnormalized.config(image="", width=0, height=0)
            self.bSaveAdversarialImage.grid_remove()

    def refresh_attack(self):
        if self.attack_on_the_fly.get() is True and (C.is_not_blank(self.target_classification.get()) or self.targeted_attack.get() is False):
            self.attack_image()

    def refresh_image(self):
        if self.img_index >= 0:
            if self.filemode.get() == False and C.is_not_blank(self.dataset_type):
                # save current img_index in a variable
                current_img_index = self.img_index
                self.open_dataset(dataset_type=self.dataset_type)
                # restores img_index after reopening the dataset
                self.img_index = current_img_index
            self.seek(img_index=self.img_index)
            if self.model is not None:
                self.refresh_classify()

    def refresh_classify(self):
        if self.filemode.get() is True and C.is_not_blank(self.class_from_dir):
            self.source_classification_fromsource.set(self.class_from_dir)
            self.source_classificationLabel_text.set("Class retrieved from filesystem: " + self.source_classification_fromsource.get())
        elif self.filemode.get() is False and self.dataset is not None:
            self.source_classification_fromsource.set(C.get_cifar10_label_from_id(self.dataset[self.img_index][1]))
            self.source_classificationLabel_text.set("Class retrieved from dataset: " + self.source_classification_fromsource.get())
        elif self.t_source_image is not None:
            self.source_classification_fromsource.set("")
            self.source_classificationLabel_text.set("Unable to determine source class.\rPlease select manually.")
        else:
            self.source_classification_fromsource.set("")
            self.source_classificationLabel_text.set("")
        
        self.set_classification()

        if self.classify_on_the_fly.get() == True:
            self.classify_image()
        else:
            self.sourceImage_prediction.set("")
            self.adversarialImage_prediction.set("")
        self.refresh_gui()

    def refresh_gui(self):
        if self.max_index >= 0:
            self.browse_frame_label_text.set(f"Browse images: {self.max_index + 1} items")
        self.img_index_human.set(self.img_index + 1)
        self.unlockbuttons()
        self.zoomer()

    def reset_gui(self):
        self.class_from_dir = None
        self.dataloader = None
        self.dataset = None
        self.PIL_source_image = None
        self.PIL_source_image_u = None
        self.PIL_delta_image = None
        self.PIL_adv_image = None
        self.PIL_adv_image_u = None
        self.t_source_image = None
        self.t_source_image_u = None
        self.t_delta_image = None
        self.t_adv_image = None
        self.t_adv_image_u = None
        self.img_index = -1
        self.max_index = -1
        self.img_index_human.set(self.img_index + 1)
        self.browse_frame_label_text.set("Browse images")
        self.img_filename_display.set("Choose a directory, a file or a dataset")
        self.source_classification.set("")
        self.source_classificationLabel_text.set("")
        self.sourceImage.config(image="", width=0, height=0)
        self.sourceImage_prediction.set("")
        self.sourceImageUnnormalized.config(image="", width=0, height=0)
        self.deltaImage.config(image="", width=0, height=0)
        self.adversarialImage.config(image="", width=0, height=0)
        self.adversarialImage_prediction.set("")
        self.adversarialImageUnnormalized.config(image="", width=0, height=0)
        self.bSaveAll.grid_remove()
        self.bSaveSourceImage.grid_remove()
        self.bSaveAdversarialImage.grid_remove()
        self.bSaveDeltaImage.grid_remove()
        self.bSaveSourceImageUnnormalized.grid_remove()
        self.bSaveAdversarialImageUnnormalized.grid_remove()
        self.bCheckOverride.grid_remove()
        self.bCheckTargetedAttack.grid_remove()
        self.mean = torch.zeros([1, 3], dtype=torch.float32)
        self.std = torch.zeros([1, 3], dtype=torch.float32)
        self.override_classification.set(False)
        self.classify_on_the_fly.set(False)
        self.attack_on_the_fly.set(False)
        self.targeted_attack_enable.set(False)
        self.targeted_attack.set(False)
        self.target_classification.set("")
        self.attacktype.set("VANILA")
        self.refresh_classify()

    def __init__(self, master=None):
        Tk.__init__(self, master)
        self.title("ML Model Toolkit")

        self.maxargs = 10
        self.max_zoom = 8
        self.img_index_human = IntVar()
        self.browse_frame_label_text = StringVar()
        self.img_filename_display = StringVar()
        self.img_filename = StringVar()
        self.source_classification = StringVar()
        self.source_classification_fromsource = StringVar()
        self.source_classification_override = StringVar()
        self.override_classification = BooleanVar()
        self.target_classification = StringVar()
        self.filemode = BooleanVar()
        self.filemode.set(True)
        self.targeted_attack = BooleanVar()
        self.use_normalization = BooleanVar()
        self.use_normalization.set(False)
        self.model = None
        self.model_filepath = StringVar()
        self.model_filepath.set("No model loaded")
        self.stats_filepath = StringVar()
        self.files = []
        self.classify_on_the_fly = BooleanVar()
        self.attack_on_the_fly = BooleanVar()
        self.attacktype = StringVar()
        self.targeted_attack_enable = BooleanVar()
        self.sourceImage_prediction = StringVar()
        self.adversarialImage_prediction = StringVar()
        self.zoomlevel = IntVar()
        self.zoomlevel.set(4)
        # Dynamic text fields
        self.source_classificationLabel_text = StringVar()
        self.class_list = C.cifar10_labels

        self.attack_list = [ "VANILA",
        "FGSM",
        "CW",
        "BIM",
        "DeepFool",
        "PGD",
        "OnePixel",
        "GN",
        "RFGSM",
        "EOTPGD",
        "FFGSM",
        "TPGD",
        "MIFGSM",
        "UPGD",
        "APGD",
        "APGDT",
        "DIFGSM",
        "TIFGSM",
        "Jitter",
        "NIFGSM",
        "PGDRS",
        "SINIFGSM",
        "VMIFGSM",
        "VNIFGSM",
        "PGDL2",
        "PGDRSL2",
        "SparseFool",
        "FAB",
        "AutoAttack",
        "Square",        
        "SPSA",
        "JSMA" ]

        # Text and button styling
        Button_style = ("Calibri", 12)
        Label_style = ("Calibri", 12, "bold")
        Classification_style = ("Calibri", 12)
        Status_bar_style = ("Calibri", 10)
        Console_style = ("Consolas", 10)
        
        # Disk icon for save buttons
        PIL_disk_icon = PIL_Image.open(C.disk_icon())
        Tk_disk_icon = PIL_ImageTk.PhotoImage(PIL_disk_icon)

        # Create main frames
        left_frame = Frame(self, width=400, bg="grey")
        left_frame.grid(row=0, rowspan = 4, column=0, padx=10, pady=5, sticky="w"+"e"+"n"+"s")
        left_frame.rowconfigure(0, weight=1)

        main_frame = Frame(self, width=850, height=185)
        main_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w"+"e"+"n"+"s")

        toolbar_frame = Frame(self, width=850, height=100)
        toolbar_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w"+"e"+"n"+"s")

        bottom_frame = Frame(self, width=850, height=200)
        bottom_frame.grid(row=2, column=1, padx=10, pady=5, sticky="w"+"e"+"n"+"s")

        status_bar = Frame(self, width=850, height=15, borderwidth=1, relief="groove")
        status_bar.grid(row=3, column=1, padx=3, pady=3, sticky="w"+"e"+"n"+"s")
        
        ##### LEFT FRAME
        self.log_text = Text(left_frame, wrap="word", width=80, font=Console_style)
        self.log_text.grid(column=0, columnspan=2, row=0, padx=5, pady=0, sticky="w"+"e"+"n"+"s")
        scrollbar = Scrollbar(left_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=2, sticky="n"+"s")
        self.log_text["yscrollcommand"] = scrollbar.set

        sys.stdout = TextRedirector(self.log_text, "stdout")

        self.bClearLog = Button(left_frame, text="Clear log", command=self.clear_log, font=Button_style)
        self.bClearLog.grid(column=0, row=1, padx=5, pady=5, sticky="w"+"s")
        self.bSaveLog = Button(left_frame, compound="left", text="Save log", image=Tk_disk_icon, command=self.save_log, font=Button_style)
        self.bSaveLog.grid(column=1, row=1, padx=5, pady=5, sticky="e"+"s")

        ###### MAIN FRAME

        # Create frames and labels in main_frame
        Label(main_frame, text="Source selection", font=Label_style, borderwidth=1, relief="ridge").grid(row=0, column=0, padx=5, pady=5)
        Label(main_frame, text="Model", font=Label_style, borderwidth=1, relief="ridge").grid(row=0, column=1, padx=5, pady=5)
        Label(main_frame, text="Attack", font=Label_style, borderwidth=1, relief="ridge").grid(row=0, column=2, padx=5, pady=5)
        
        # image_menu
        self.image_menu = Frame(main_frame, height=185, bg="lightgrey")
        self.model_menu = Frame(main_frame, height=185, bg="lightgrey")
        self.attack_menu = Frame(main_frame, height=185, bg="lightgrey")
        self.browse_frame = Frame(self.image_menu, height=30, bg="lightgrey")
        self.zoom_frame = Frame(self.image_menu, height=30, bg="lightgrey")

        self.image_menu.grid(row=2, column=0, padx=5, pady=5, sticky="n")
        self.model_menu.grid(row=2, column=1, padx=5, pady=5, sticky="n")
        self.attack_menu.grid(row=2, column=2, padx=5, pady=5, sticky="n")
        
        self.bFileMode1 = Radiobutton(self.image_menu, text="File mode", variable=self.filemode, value=True, font=Button_style)
        self.bFileMode2 = Radiobutton(self.image_menu, text="Dataset mode", variable=self.filemode, value=False, font=Button_style)
        self.bOpenFile = Button(self.image_menu, text="Open File", command=self.open_file, width=15, font=Button_style)
        self.bOpenDir = Button(self.image_menu, text="Open Folder", command=self.open_dir, width=15, font=Button_style)
        self.bOpenTrainDataset = Button(self.image_menu, text="CIFAR-10\nTrain Dataset", command=lambda *e: self.open_dataset(dataset_type="train"), font=Button_style)
        self.bOpenTestDataset = Button(self.image_menu, text="CIFAR-10\nTest Dataset", command=lambda *e: self.open_dataset(dataset_type="test"), font=Button_style)

        self.bFileMode1.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.bFileMode2.grid(row=0, column=1, padx=5, pady=10, sticky="e")
        self.bOpenFile.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bOpenDir.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bOpenTrainDataset.grid(row=3, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bOpenTestDataset.grid(row=3, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")

        # browse_frame
        self.browse_frame.grid(row=4, column=0, columnspan=2, padx=5, sticky="n")
        self.browse_frame_label = Label(self.browse_frame, textvariable=self.browse_frame_label_text, font=Label_style, borderwidth=1, relief="ridge")
        self.bPrev = Button(self.browse_frame, text="Prev", command=self.seek_prev, width=5, state="disabled", font=Button_style)
        self.index_check = Entry(self.browse_frame, textvariable=self.img_index_human, state="disabled", width=10, font=Label_style)
        self.bNext = Button(self.browse_frame, text="Next", command=self.seek_next, width=5, state="disabled", font=Button_style)
        
        self.browse_frame_label.grid(row=0, column=0, columnspan=3, padx=5)
        self.bPrev.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.index_check.grid(row=1, column=1, padx=5, pady=5)
        self.bNext.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        
        self.index_check.bind("<Return>", lambda *e: self.seek_to())
        self.index_check.grid()

        # zoom_frame
        self.zoom_frame.grid(row=5, column=0, columnspan=2, padx=5, sticky="n")
        Label(self.zoom_frame, text="Zoom", font=Label_style, borderwidth=1, relief="ridge").grid(row=0, column=0, columnspan=3, padx=5)
        self.bZoomDown = Button(self.zoom_frame, text="-", command=self.zoomdown, width=5, state="disabled", font=Button_style)
        self.bZoomDown.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.zoomlevel_label = Label(self.zoom_frame, textvariable=str(self.zoomlevel), font=Label_style)
        self.zoomlevel_label.grid(row=1, column=1, padx=5, pady=5)
        self.bZoomUp = Button(self.zoom_frame, text="+", command=self.zoomup, width=5, state="disabled", font=Button_style)
        self.bZoomUp.grid(row=1, column=2, padx=5, pady=5, sticky="e")

        # model_menu
        self.model_filepathlabel = Label(self.model_menu, textvariable=str(self.model_filepath))
        self.model_filepathlabel.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

        self.bLoadModel = Button(self.model_menu, text="Load model", command=self.open_model, font=Button_style)
        self.bLoadModel.grid(row=2, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bUseNormalization = Checkbutton(self.model_menu, text="Source/Model is\rusing Normalization", variable=self.use_normalization, font=Button_style)
        self.bUseNormalization.grid(row=2, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")

        self.bShowTrainingCurves = Button(self.model_menu, text="Show training curves", command=self.show_training_curves, state="disabled", font=Button_style)
        self.bShowTrainingCurves.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")

        self.bClassifyDataset = Button(self.model_menu, text="Classify dataset", command=self.classify_dataset, state="disabled", font=Button_style)
        self.bClassifyDataset.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bClassifyAll = Button(self.model_menu, text="Classify all images", command=self.recognize_image_seek, state="disabled", font=Button_style)
        self.bClassifyAll.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bClassify = Button(self.model_menu, text="Classify image", command=self.classify_image, state="disabled", font=Button_style)
        self.bClassify.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bCheckClassify = Checkbutton(self.model_menu, text="Classify on the fly", variable=self.classify_on_the_fly, onvalue=True, offvalue=False, state="disabled", font=Button_style)
        self.bCheckClassify.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="w"+"e"+"n"+"s")


        # attack_menu
        attack_type_menu = OptionMenu(self.attack_menu, self.attacktype, *self.attack_list)
        attack_type_menu.grid(row=1, column=0, padx=5, pady=10, sticky="w"+"e"+"n"+"s")

        self.attack_args_menu = []
        self.attack_args_values = []
        self.attack_args_labels = []
        self.attack_args_names = []

        for i in range(self.maxargs):
            self.attack_args_values.insert(i, StringVar())
            self.attack_args_names.insert(i, StringVar())
            self.attack_args_labels.insert(i, Label(self.attack_menu, textvariable=self.attack_args_names[i]))
            self.attack_args_labels[i].grid(row=2, column=i, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.attack_args_labels[i].grid_remove()
            self.attack_args_menu.insert(i, Entry(self.attack_menu, textvariable=self.attack_args_values[i], width=10))
            self.attack_args_menu[i].grid(row=3, column=i, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
            self.attack_args_menu[i].grid_remove()

        self.bAttackDataset = Button(self.attack_menu, text="Attack dataset", command=self.attack_dataset, state="disabled", font=Button_style)
        self.bAttackDataset.grid(row=4, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bAttackAll = Button(self.attack_menu, text="Attack all images", command=self.attack_image_seek, state="disabled", font=Button_style)
        self.bAttackAll.grid(row=5, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bAttack = Button(self.attack_menu, text="Attack image", command=self.attack_image, state="disabled", font=Button_style)
        self.bAttack.grid(row=6, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bCheckAttack = Checkbutton(self.attack_menu, text="Attack on the fly", variable=self.attack_on_the_fly, onvalue=True, offvalue=False, state="disabled", font=Button_style)
        self.bCheckAttack.grid(row=7, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        
        ##### TOOLBAR FRAME
        toolbar_source_frame = Frame(toolbar_frame, width=300, height=50)
        toolbar_delta_frame = Frame(toolbar_frame, width=150, height=50)
        toolbar_adv_frame = Frame(toolbar_frame, width=300, height=50)
        toolbar_source_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w"+"e"+"n"+"s")
        toolbar_delta_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w"+"e"+"n"+"s")
        toolbar_adv_frame.grid(row=0, column=2, padx=10, pady=5, sticky="w"+"e"+"n"+"s")

        # toolbar_source_frame
        toolbar_source_frame.columnconfigure(index=0, weight=1)
        self.source_classificationLabel = Label(toolbar_source_frame, textvariable=str(self.source_classificationLabel_text), font=Classification_style)
        self.source_classificationLabel.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.bCheckOverride = Checkbutton(toolbar_source_frame, text="Override classification", variable=self.override_classification, onvalue=True, offvalue=False, state="disabled", font=Button_style)
        self.bCheckOverride.grid(row=1, column=0, padx=5, pady=10, sticky="s"+"w")
        
        self.override_class_menu = OptionMenu(toolbar_source_frame, self.source_classification_override, *self.class_list)
        self.override_class_menu.grid(row=1, column=1, padx=5, pady=5, sticky="s"+"w")
        
        # toolbar_delta_frame
        self.bSaveAll = Button(toolbar_delta_frame, compound="left", text="Save all\rimages", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="grid"), font=Button_style)
        self.bSaveAll.grid(row=0, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        toolbar_delta_frame.columnconfigure(index=0, weight=1)
        toolbar_delta_frame.rowconfigure(index=0, weight=1)
        self.bSaveAll.image = Tk_disk_icon

        # toolbar_adv_frame
        self.bCheckTargetedAttack = Checkbutton(toolbar_adv_frame, text="Targeted attack", variable=self.targeted_attack, onvalue=True, offvalue=False, state="disabled", font=Button_style)
        self.bCheckTargetedAttack.grid(row=1, column=0, padx=5, pady=10, sticky="s"+"w")
        
        toolbar_adv_frame.columnconfigure(index=0, weight=1)
        toolbar_adv_frame.rowconfigure(index=0, weight=1)
        self.targeted_class_menu = OptionMenu(toolbar_adv_frame, self.target_classification, *self.class_list)
        self.targeted_class_menu.grid(row=1, column=1, padx=5, pady=5, sticky="s"+"e")
        self.targeted_class_menu.config(state="disabled")
        
        
        ##### BOTTOM FRAME
        source_frame = Frame(bottom_frame, width=350, height=200)
        delta_frame = Frame(bottom_frame, width=150, height=200)
        adversarial_frame = Frame(bottom_frame, width=350, height=200)
        source_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n"+"w")
        delta_frame.grid(row=0, column=1, padx=10, pady=5, sticky="n")
        adversarial_frame.grid(row=0, column=2, padx=10, pady=5, sticky="n"+"e")

        # source_frame
        self.sourceImageUnnormalized = Label(source_frame, borderwidth=1)
        self.sourceImageUnnormalized.grid(row=1, column=0, padx=5, pady=5)
        self.sourceImage = Label(source_frame, borderwidth=1)
        self.sourceImage.grid(row=1, column=1, padx=5, pady=5)
        self.sourceImage_predictionlabel = Label(source_frame, textvariable=str(self.sourceImage_prediction), font=Classification_style)
        self.sourceImage_predictionlabel.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        self.bSaveSourceImage = Button(source_frame, compound="left", text="Source", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="source"), font=Button_style)
        self.bSaveSourceImage.grid(row=0, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bSaveSourceImageUnnormalized = Button(source_frame, compound="left", text="Source (Unnormalized)", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="source_u"), font=Button_style)
        self.bSaveSourceImageUnnormalized.grid(row=0, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bSaveSourceImage.image = Tk_disk_icon
        self.bSaveSourceImageUnnormalized.image = Tk_disk_icon

        # delta_frame
        self.deltaImage = Label(delta_frame, borderwidth=1)
        self.deltaImage.grid(row=1, column=0, padx=5, pady=5)
        self.bSaveDeltaImage = Button(delta_frame, compound="left", text="Delta image", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="delta"), font=Button_style)
        self.bSaveDeltaImage.grid(row=0, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bSaveDeltaImage.image = Tk_disk_icon

        # adversarial_frame
        self.adversarialImage = Label(adversarial_frame, borderwidth=1)
        self.adversarialImage.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.adversarialImageUnnormalized = Label(adversarial_frame, borderwidth=1)
        self.adversarialImageUnnormalized.grid(row=1, column=1, padx=5, pady=5)
        self.adversarialImage_predictionlabel=Label(adversarial_frame, textvariable=str(self.adversarialImage_prediction), font=Classification_style)
        self.adversarialImage_predictionlabel.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.bSaveAdversarialImage = Button(adversarial_frame, compound="left", text="Adversarial", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="adversarial"), font=Button_style)
        self.bSaveAdversarialImage.grid(row=0, column=0, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bSaveAdversarialImageUnnormalized = Button(adversarial_frame, compound="left", text="Adversarial (unnormalized)", image=Tk_disk_icon, command=lambda *e: self.save_file(imagetype="adversarial_u"), font=Button_style)
        self.bSaveAdversarialImageUnnormalized.grid(row=0, column=1, padx=5, pady=5, sticky="w"+"e"+"n"+"s")
        self.bSaveAdversarialImage.image = Tk_disk_icon
        self.bSaveAdversarialImageUnnormalized.image = Tk_disk_icon
        
        # status_bar
        self.img_filenamelabel = Label(status_bar, textvariable=str(self.img_filename_display), font=Status_bar_style)
        self.img_filenamelabel.grid(row=0, column=0, padx=2, pady=2)
        
        # Variable tracers
        self.filemode.trace_add("write", lambda *e: self.reset_gui())
        self.img_filename.trace_add("write", lambda *e: self.unlockbuttons())
        self.source_classification_override.trace_add("write", lambda *e: self.refresh_classify())
        self.override_classification.trace_add("write", lambda *e: self.refresh_classify())
        self.targeted_attack.trace_add("write", lambda *e: self.refresh_gui())
        self.target_classification.trace_add("write", lambda *e: self.refresh_attack())
        self.model_filepath.trace_add("write", lambda *e: self.refresh_gui())
        self.classify_on_the_fly.trace_add("write", lambda *e: self.refresh_classify())
        self.attack_on_the_fly.trace_add("write", lambda *e: self.refresh_attack())
        self.zoomlevel.trace_add("write", lambda *e: self.refresh_gui())
        self.attacktype.trace_add("write", lambda *e: self.attack_inspect())
        self.use_normalization.trace_add("write", lambda *e: self.refresh_image())

        # Select CUDA if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.batch_size = 100
        else:
            self.device = torch.device("cpu")
            self.batch_size = 50

        print(f"Device in use: {str(self.device)}")

        self.reset_gui()
        self.attack_inspect()
        
if __name__ == "__main__":
    app = App(); app.mainloop()