import torch
from common_functions import train_model, test_model, test_classes, save_model, load_model, load_model_stats, save_model_stats, create_dataloader, plot_training_results

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {str(device)}")
    if str(device) == "cpu":
        batch_size = 50
    else:
        batch_size = 100
    train_loader, validate_loader = create_dataloader(train=True, batch_size=batch_size, normalize=False)
    test_loader = create_dataloader(train=False, batch_size=batch_size, normalize=False)

    num_classes = len(test_loader.dataset.classes)

    from cnn import *
    model1 = CNN1(num_classes)
    model2 = CNN2(num_classes)
    model3 = CNN3(num_classes)
    model4 = CNN4(num_classes)

    train_model(model1, num_epochs=50, device=device, model_name="CNN1", train_dataloader=train_loader, validation_dataloader=validate_loader, prevent_overfit=False, save_stats=True)
    #model1 = load_model("CNN1.pth", device)
    plot_training_results(*load_model_stats("CNN1.pkl"))
    test_model(model=model1, device=device, dataloader=test_loader) 
    test_classes(model=model1, device=device, dataloader=test_loader)

    train_model(model2, num_epochs=50, device=device, model_name="CNN2", train_dataloader=train_loader, validation_dataloader=validate_loader, prevent_overfit=False, save_stats=True)
    #model2 = load_model("CNN2.pth", device)
    plot_training_results(*load_model_stats("CNN2.pkl"))
    test_model(model=model2, device=device, dataloader=test_loader) 
    test_classes(model=model2, device=device, dataloader=test_loader)

    #train_model(model3, num_epochs=30, device=device, model_name="CNN3", train_dataloader=train_loader, validation_dataloader=validate_loader, prevent_overfit=False, save_stats=True)
    model3 = load_model("CNN3.pth", device)
    plot_training_results(*load_model_stats("CNN3.pkl"))
    test_model(model=model3, device=device, dataloader=test_loader) 
    test_classes(model=model3, device=device, dataloader=test_loader)

    #train_model(model4, num_epochs=30, device=device, model_name="CNN4", train_dataloader=train_loader, validation_dataloader=validate_loader, prevent_overfit=False, save_stats=True)
    model4 = load_model("CNN4.pth", device)
    plot_training_results(*load_model_stats("CNN4.pkl"))
    test_model(model=model4, device=device, dataloader=test_loader) 
    test_classes(model=model4, device=device, dataloader=test_loader)

    #model_pret_1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a2", pretrained=True, trust_repo=True).eval()
    #model_pret_2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5", pretrained=True, trust_repo=True).eval()
    #print(model_pret_1)
    #print(model_pret_2)

    #test_model(model_pret_1, device)
    #test_classes(model_pret_1, device)
    #test_model(model_pret_2, device)
    #test_classes(model_pret_2, device)
