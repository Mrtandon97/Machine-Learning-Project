import argparse
import os

from ml4gpr import feature_extraction, hyperbola_detection, preprocess

def run(data_dir, model_path):
    while True:
        data_listdir = os.listdir(data_dir)
        if data_listdir:
            cropped_data = preprocess.crop_data(data_listdir[0])

            features = feature_extraction.extract_features(cropped_data)

            pred = hyperbola_detection.run_inference(model_path, features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')

    args = parser.parse_args()
    print(args)

    # do some error handling here
    data_dir = args.data_dir
    model_path = args.model_path

    #run()