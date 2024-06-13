import h5py

def load_h5_file(filepath):
    with h5py.File(filepath, 'r') as file:
        # 打印文件中的所有键
        keys = list(file.keys())
        print(f"Keys in {filepath}: {keys}")

        # 假设文件中只有一个数据集，并读取第一个键的数据
        if keys:
            key = keys[1]
            data = file[key][:]
            return data
        else:
            raise KeyError(f"No datasets found in {filepath}")

# 示例加载某个测试文件，并打印文件中的键
test_data_example = load_h5_file('./data_set/Synapse/test_vol_h5/case0001.npy.h5')
print(test_data_example.shape)
