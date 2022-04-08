import fiftyone as fo
import argparse

def main(arg):
    data_dir = '/opt/ml/detection/dataset/'
    # anno_dir = '/opt/ml/detection/dataset/' + arg.data_dir + '.json'
    anno_dir = arg.data_dir
    print(data_dir)
    print(anno_dir)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_dir,
        labels_path=anno_dir,
    )
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 'stratified_kfold/basic_v2/cv_val_3'
    parser.add_argument('--data_dir', '-d', type=str, default='single_inf.json', 
                        help='imageData directory: "trian" or "test". default is "train"')
    parser.add_argument('--port', '-p', type=int, default=30001,
                        help='Port Number')
    args = parser.parse_args()
    main(args)