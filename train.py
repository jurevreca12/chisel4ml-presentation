import torch
from torch.nn.utils import prune
import brevitas.nn as qnn

def print_sparsity(model):
    global_zeros = 0
    global_elems = 0
    for module_name, module in model.named_modules():
        if hasattr(module, 'weight'):
            num_zeros = torch.sum(module.weight == 0)
            num_elems = module.weight.shape.numel()
            global_zeros += num_zeros
            global_elems += num_elems
            prune_ratio = 100. * (float(num_zeros) / float(num_elems))
            print(f"Sparsity in {module_name}.weight: {prune_ratio:.2f}%.")

    global_prune_ratio = 100. * (float(global_zeros) / float(global_elems))
    print(f"Global sparsity is: {global_prune_ratio:.2f}%.")


def prune_model_global_unstructured(model, prune_rate, print_sparsity=False):
    parameters_to_prune = []
    for module in model.modules():
        if hasattr(module, 'weight'):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_rate
    )

    # Remove prunning re-parameterization
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")


def train_model(model, train_loader, criterion, optimizer, epochs, device, prune_rate):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            prune_model_global_unstructured(model, prune_rate)

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:
                print(
                    f"[{epoch + 1}/{epochs}, {i + 1:3d}/{len(train_loader)}]"
                    f" - loss: {running_loss / 2000:.5f}"
                )
                running_loss = 0.0
    print("Finished Training")


def eval_model(model, test_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = float(correct) / total
    print(f"Accuracy of the network on the {len(test_loader)} test input batches: {100 * accuracy} %")
    return accuracy


def merge_batchnorm(act, bn, ltype="conv"):
    if ltype == "conv":
        w_act = act.weight.clone().view(act.out_channels, -1)
    else:
        w_act = act.weight.clone()
    inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w_bn = torch.diag(inv)
    new_w = torch.mm(w_bn, w_act).view(act.weight.size())
    if act.bias is not None:
        b_act = act.bias
    else:
        b_act = torch.zeros(act.weight.size(0))
    new_b = ((b_act - bn.running_mean) * inv) + bn.bias
    act.weight = torch.nn.Parameter(new_w)
    act.bias = torch.nn.Parameter(new_b)


def train_quant_model(model, model_nobn, train_loader, test_loader, bitwidth, prune_rate, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        epochs=epochs,
        device=device,
        prune_rate=prune_rate,
    )
    print_sparsity(model)

    print(f"ACCURACY WITH BN {bitwidth}:")
    eval_model(model, test_loader, device)
    print("MERGING BATCHNORM TO ACTIVE LAYERS")
    model.eval()
    model_nobn.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "bn" not in k}
    )
    if hasattr(model, 'conv0'):  # cnn_model
        for layer in (model.conv0, model.conv1):
            merge_batchnorm(layer.conv, layer.bn)
        for layer in (model.dense0, model.dense1):
            merge_batchnorm(layer.dense, layer.bn)
    else:  # lhc_jets model
        lin_bn_pairs = (
            (model.linear0, model.bn0),
            (model.linear1, model.bn1),
            (model.linear2, model.bn2),
            (model.linear3, model.bn3),
        )
        for lin, bn in lin_bn_pairs:
            merge_batchnorm(lin, bn)

    model_nobn.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "bn" not in k}, strict=False
    )
    print("BN layers fused.")
    print("RETRAINING FUSED MODEL")
    train_model(
        model=model_nobn,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model_nobn.parameters(), lr=0.001),
        epochs=epochs,
        device=device,
        prune_rate=prune_rate,
    )
    print_sparsity(model)
    print(f"FINAL ACCURACY {bitwidth} (NO BN):")
    final_acc = eval_model(model_nobn, test_loader, device)
    # return trained model and one batch of data for testing
    torch_tensor = next(iter(test_loader))[0]
    return model_nobn, torch_tensor.detach().numpy(), final_acc