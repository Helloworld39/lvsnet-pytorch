from torch.utils.data import TensorDataset, DataLoader


def data_loader(*tsr):
    def f(batch_size=16, is_shuffled=False):
        dataset = TensorDataset(*tsr)
        if is_shuffled:
            return DataLoader(dataset, batch_size, True, drop_last=True)
        else:
            return DataLoader(dataset, batch_size, False)
    return f
