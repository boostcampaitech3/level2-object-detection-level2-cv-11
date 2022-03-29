import fiftyone as fo
import argparse

def main(arg):
    data_dir = '/opt/ml/detection/dataset/'
    anno_dir = args.anno_dir
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='train', # test??/
                        help='imageData directory: "train" or "test". default is "train"')
    parser.add_argument('--port', '-p', type=int, default=30006,
                        help='Port Number')
    parser.add_argument('--anno_dir', '-a', type=str, default='./csv_to_json.json')
    args = parser.parse_args()
    main(args)
