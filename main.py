import data
import model
import argparse


def main(data_path):
    dataset_en, unknown_en = data.load_dataset(data_path + "/dataset_en.pkl")
    dataset_vi, unknown_vi = data.load_dataset(data_path + "/dataset_vi.pkl")
    dict_en, dict_vi = data.load_dict(data_path)
    print(dataset_en[1])
    # model.train((dict_en, dict_vi), (dataset_en, unknown_en), (dataset_vi, unknown_vi))



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--path', type=str, default='./data')

    args = parser.parse_args()

    main(args.path)
