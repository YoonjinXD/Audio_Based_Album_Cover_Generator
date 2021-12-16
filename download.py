import os
import numpy as np
import pandas as pd
import requests
import csv

multi_data = pd.read_csv("./MuMu_dataset/MuMu_dataset_multi-label.csv", sep=",", index_col=0)
amazon_data = pd.read_json("./MuMu_dataset/amazon_metadata_MuMu.json")

img_path = "./MuMu_dataset/album_imgs"
csv_path = "./MuMu_dataset/rearranged_MuMu_dataset.csv"

headers = ['amazon_id', 'album_img_path', 'MSD_track_id']

total_data_num = len(multi_meta)
no_match_error = 0
req_error = 0
no_img_error = 0

print("Start download #%d of album images" %(total_data_num))
if not os.path.exists(img_path):
    os.makedirs(img_path)

with open(csv_path, 'w', encoding='UTF8') as c:
    writer = csv.writer(c)
    writer.writerow(headers)

    for idx in range(len(multi_data)):
        # iteration checking
        if idx % 5000 == 0:
            print("Downloaing...", int(idx/total_data_num*100), "%")

        track_data = multi_data.iloc[idx]
        amazon_id = track_data.name
        
        # check valid match
        try:
            # TODO: fix this!
            img_url = amazon_meta.loc[amazon_id].values[0][3]
        except:
            no_match_error += 1
            continue

        # request
        r = requests.get(img_url)
        if r.status_code != 200:
            req_error += 1
            continue

        # check no-img-lg
        _, f_name = os.path.split(img_url)
        if "no-img-lg" in f_name or ".gif" in f_name:
            no_img_error += 1
            continue

        # save image
        img_path = os.path.join(img_path, amazon_id + ".jpg")
        with open(img_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
        f.close()

        # Write csv
        msd_track_id = track_data['MSD_track_id']
        row = [amazon_id, img_path, msd_track_id]
        writer.writerow(row)

    c.close()  
        
print("Completed.")
print("From total #%d of data,\nNo matching error: %d\nRequest error: %d\nNo image error: %d" %(total_data_num, no_match_error, req_error_error, no_image_error))