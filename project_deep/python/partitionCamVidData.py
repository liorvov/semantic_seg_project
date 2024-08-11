import torch

# Partition CamVid data by randomly selecting 60% of the data for training, 20% for validation, and 20% for testing.

def partitionCamVidData(df):

    train_full_size = 701 - 141
    test_full_size = len(df) - train_full_size
    train_data_full, test_data_full = torch.utils.data.random_split(df, [train_full_size, test_full_size])
    train_size = 420
    validation_size = len(train_data_full) - train_size
    train_data, validation_data = torch.utils.data.random_split(train_data_full, [train_size, validation_size])
    unused_size = 1
    test_size = test_full_size - unused_size
    test_data, unused_data = torch.utils.data.random_split(test_data_full, [test_size, unused_size])

    return train_data, validation_data, test_data
