from camvidPixelLabelIDs import *
from countEachLabel import *
from partitionCamVidData import *
from outputToLabel import *
from labelToColormap import *
from validate import *
from calc_accuracy import *
from torch.utils.data import Dataset
from segmentAndShow import *
from LSTMModel import *
from LSTMDataSet import *
import os
import requests
import zipfile
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from nn_eff import *
import segmentation_models_pytorch as smp


def main():

    # define run settings:
    # choose network:
    use_resent50 = True
    use_efficientnet_v2 = False
    use_efficientnet_b4 = False

    # choose hyper parameters:
    initial_lr = 1e-4
    momentum_for_sgd = 0.9
    regularization_factor = 0.0005
    scheduler_factor = 0.3
    scheduler_milestones = [10, 20, 30]
    lstm_initial_lr = 0.012
    lstm_scheduler_factor = 0.7
    lstm_scheduler_milestones = [1, 2]
    epochs = 30
    lstm_epochs = 3
    batch_size = 4
    use_adam = True
    test = True
    train = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = 720
    width = 960

    # downloading data:
    tempdir = os.getcwd()
    image_url = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip'
    label_url = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip'

    output_folder = os.path.join(tempdir, 'CamVid')
    labels_zip = os.path.join(output_folder, 'labels.zip')
    images_zip = os.path.join(output_folder, 'images.zip')

    if (not os.path.exists(labels_zip)) or (not os.path.exists(images_zip)):
        os.mkdir(output_folder)

        # downloading images:
        print('Downloading 16 MB CamVid dataset images...')
        response = requests.get(image_url)
        open(images_zip, "wb").write(response.content)
        with zipfile.ZipFile(images_zip, "r") as zip_ref:
            zip_ref.extractall(os.path.join(output_folder, 'images'))

        # downloading labels:
        print('Downloading 16 MB CamVid dataset labels...')
        response = requests.get(label_url)
        open(labels_zip, "wb").write(response.content)
        with zipfile.ZipFile(labels_zip, "r") as zip_ref:
            zip_ref.extractall(os.path.join(output_folder, 'labels'))

    # spliting 30 original classes over 11 new classes:
    classes = [
        "Sky",
        "Building",
        "Pole",
        "Road",
        "Pavement",
        "Tree",
        "SignSymbol",
        "Fence",
        "Car",
        "Pedestrian",
        "Bicyclist"
    ]
    label_ids = camvid_pixel_label_ids()
    reversed_label_ids = reverse_label_ids(label_ids)

    label_ids_list_r = []
    label_ids_list_g = []
    label_ids_list_b = []
    for val in list(label_ids.values()):
        label_ids_list_r.append(val[0][0])
        label_ids_list_g.append(val[0][1])
        label_ids_list_b.append(val[0][2])
    # list all files:
    img_dir = os.path.join(output_folder, 'images', '701_StillsRaw_full')
    label_dir = os.path.join(output_folder, 'labels')

    data = myDataset(img_dir, label_dir)

    # analyze dataset statistics:
    tbl = count_each_label(data, label_ids)
    classified_pixels_num = sum(tbl["PixelCount"])
    tbl["frequency"] = tbl["PixelCount"]/classified_pixels_num
    tbl["class weights"] = np.median(tbl["frequency"]) / tbl["frequency"]
    class_weights = torch.tensor(tbl["class weights"].values).float().to(device)
    print(tbl)

    # split data into train, validation and test:
    train_data, validation_data, test_data = partitionCamVidData(data)
    num_training_images = len(train_data)
    num_val_images = len(validation_data)
    num_testing_images = len(test_data)
    print('Train data size:', num_training_images, '\nValidation data size:',  num_val_images, '\nTest data size:',
          num_testing_images)

    # creating the network:
    num_classes = len(classes)
    if use_resent50:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', num_classes=num_classes).to(device)
        model_name = 'resnet50'
    elif use_efficientnet_v2:
        model = DeepLabV3(num_classes).to(device)
        model_name = 'efficientnet_v2'
        # model = effnetv2_s(num_classes=num_classes, width=width, height=height).to(device)
    elif use_efficientnet_b4:
        model = smp.DeepLabV3Plus(encoder_name="efficientnet-b4", in_channels=3, classes=num_classes).to(device)
        model_name = 'efficientnet_b4'
    else:
        print("Error: unrecognized model!")
        return
    # setting hyper-parameters
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=regularization_factor)
    else:
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum_for_sgd,
                              weight_decay=regularization_factor)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_factor)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    valid_batch_size = 2
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=valid_batch_size, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=valid_batch_size, num_workers=0)
    writer = SummaryWriter()
    batches_in_epoch = np.ceil(num_training_images / batch_size)

    input_dim = num_classes
    hidden_dim = 75
    output_dim = num_classes
    layer_dim = 2
    lstm_model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    # optimizer - the relevant uncommented. Chosen optimizer - ADAM
    # lstm_optimizer = optim.SGD(lstm_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_initial_lr)
    lstm_scheduler = optim.lr_scheduler.MultiStepLR(lstm_optimizer, milestones=lstm_scheduler_milestones,
                                                    gamma=lstm_scheduler_factor)

    model_params = sum(p.numel() for p in model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"Number of parameters in model: {model_params}")
    print(f"Number of parameters in lstm: {lstm_params}")

    # train the model
    if train:
        for e in range(epochs):
            minibatch = 0
            print(f"Epoch {e}:")
            for img, lbl in dataloader:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                img = img.permute(0, 3, 2, 1).float().to(device)
                out = model(img)["out"] if use_resent50 else model(img)
                current_batch_size = len(out)
                output = out.permute(0, 3, 2, 1).float()
                label_colormap = label_to_colormap(lbl, current_batch_size, reversed_label_ids, classes).to(device)
                output_colormap = label_to_colormap(output_to_label(output, label_ids_list_r, label_ids_list_g,
                                                                    label_ids_list_b, batch_size),
                                                    batch_size, reversed_label_ids, classes).to(device)
                loss = loss_func(output.reshape(current_batch_size * height * width, num_classes),
                                 label_colormap.view(current_batch_size * height * width).long())
                loss.backward()
                optimizer.step()
                loss = float(torch.sum(loss).data.cpu())
                accuracy = calc_accuracy(label_colormap, output_colormap)
                writer.add_scalar('training loss', loss, minibatch + e * batches_in_epoch)
                writer.add_scalar('training accuracy', accuracy, minibatch + e * batches_in_epoch)
                print(f"Minibatch {minibatch}: loss = {loss:.3f}, accuracy = {accuracy*100:.3f}%")
                minibatch += 1
                del img
                del lbl
                del out
                del output
                del label_colormap
                del output_colormap
                del loss
            scheduler.step()
            validate(model, valid_dataloader, valid_batch_size, reversed_label_ids, classes, label_ids_list_r,
                     label_ids_list_g, label_ids_list_b, num_val_images, e+1, writer, "validation", use_resent50)

        lstm_train_data = LSTMDataset(model, dataloader, use_resent50, batch_size, num_classes, reversed_label_ids, classes)
        lstm_dataloader = torch.utils.data.DataLoader(lstm_train_data, batch_size=batch_size, num_workers=0, shuffle=True)
        lstm_validation_data = LSTMDataset(model, valid_dataloader, use_resent50, valid_batch_size, num_classes,
                                           reversed_label_ids, classes)
        lstm_valid_dataloader = torch.utils.data.DataLoader(lstm_validation_data, batch_size=valid_batch_size,
                                                            num_workers=0, shuffle=True)
        del dataloader
        del valid_dataloader
        for e in range(lstm_epochs):
            minibatch = 0
            print(f"Epoch {e}:")
            for img, lbl in lstm_dataloader:
                torch.cuda.empty_cache()
                lstm_optimizer.zero_grad()
                lbl = lbl.long().to(device)
                lstm_batch_loss = 0
                lstm_batch_accuracy = 0
                for i in range(batch_size):
                    lstm_optimizer.zero_grad()
                    lstm_input = img[i].unsqueeze(0).to(device)
                    lstm_out = lstm_model(lstm_input).squeeze(0)
                    lstm_loss = loss_func(lstm_out, lbl[i])
                    lstm_loss.backward()
                    lstm_optimizer.step()
                    lstm_result = torch.argmax(lstm_out, dim=1)
                    lstm_batch_accuracy += calc_accuracy(lbl[i], lstm_result) * 100 / batch_size
                    lstm_batch_loss = lstm_batch_loss + lstm_loss
                    del lstm_input
                    del lstm_out
                    del lstm_result
                del img
                del lbl
                lstm_batch_loss = float(torch.sum(lstm_batch_loss).data.cpu())
                writer.add_scalar('lstm training loss', lstm_batch_loss, minibatch + e * batches_in_epoch)
                writer.add_scalar('lstm training accuracy', lstm_batch_accuracy, minibatch + e * batches_in_epoch)
                print(f"Minibatch {minibatch} (lstm): loss = {lstm_batch_loss:.3f}, accuracy = {lstm_batch_accuracy:.3f}%")
                minibatch += 1
            lstm_scheduler.step()
            validate_lstm(model, lstm_model, lstm_valid_dataloader, valid_batch_size, reversed_label_ids, classes,
                          label_ids_list_r, label_ids_list_g, label_ids_list_b, num_val_images, e+1, writer,
                          "validation", use_resent50)

        # save model:
        torch.save(model.state_dict(), '{0}_model_weights.pth'.format(model_name))
        torch.save(lstm_model.state_dict(), '{0}_lstm_model_weights.pth'.format(model_name))

    # test all test set:
    if test:
        lstm_test_data = LSTMDataset(model, test_dataloader, use_resent50, valid_batch_size, num_classes,
                                     reversed_label_ids, classes)
        lstm_test_dataloader = torch.utils.data.DataLoader(lstm_test_data, batch_size=valid_batch_size,
                                                           num_workers=0, shuffle=True)
        model.load_state_dict(torch.load('{0}_model_weights.pth'.format(model_name)))
        model.eval()
        lstm_model.load_state_dict(torch.load('{0}_lstm_model_weights.pth'.format(model_name)))
        lstm_model.eval()
        validate(model, test_dataloader, valid_batch_size, reversed_label_ids, classes, label_ids_list_r,
                 label_ids_list_g, label_ids_list_b, num_testing_images, 0, writer, "test", use_resent50)
        validate_lstm(model, lstm_model, lstm_test_dataloader, valid_batch_size, reversed_label_ids, classes,
                      label_ids_list_r, label_ids_list_g, label_ids_list_b, num_val_images, 0, writer,
                      "test", use_resent50)
        # test one random image from test set:
        test_img = test_dataloader.dataset[0][0]
        lbl_orig = test_dataloader.dataset[0][1]
        segment_and_show(test_img, lbl_orig, model, lstm_model, label_ids_list_r, label_ids_list_g, label_ids_list_b,
                         use_resent50, reversed_label_ids, classes, label_ids, height, width)


if __name__ == '__main__':
    main()
