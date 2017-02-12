class MyDataset(torch.utils.Dataset):
    def __init__(self):
        self.data_files = os.listdir('data_dir')
        sort(self.data_files)

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)


dset = MyDataset()
loader = torch.utils.DataLoader(dset, num_workers=8)



def mean_standardise(x)
	return transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 1, 1, 1 ])