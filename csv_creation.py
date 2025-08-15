import os
import csv

datasetDir = "C:\\Users\\Dimithri\\Documents\\AlteredDataset"
output_csv = "car_labels.csv"


progress = 0
imageCount = 1451890
# for make in os.listdir(datasetDir):
#     for model in os.listdir(os.path.join(datasetDir, make)):
#         for year in os.listdir(os.path.join(datasetDir, make, model)):
#             for item in os.listdir(os.path.join(datasetDir, make, model, year)):
#                 if os.path.isdir(os.path.join(datasetDir, make, model, year, item)):
#                     for image in os.listdir(os.path.join(datasetDir, make, model, year, item)):
#                         imageCount += 1
                
#                 else:
#                     imageCount += 1

#                 print("Image Count: ", imageCount)


rows = []
output = "car_labels.csv"
for make in os.listdir(datasetDir):
    make_path = os.path.join(datasetDir, make)


    for model in os.listdir(make_path):
        model_path = os.path.join(make_path, model)


        for year in os.listdir(model_path):
            year_path = os.path.join(model_path, year)


            label = f"{make}_{model}_{year}".replace(" ", "_")

            for image_file in os.listdir(year_path):
                image_path = os.path.join(make, model, year, image_file)
                rows.append([image_path, label])
                progress += 1
                print(f"Creating Labels: {(progress/imageCount)*100:.2f}%")

progress = 0
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    writer.writerows(rows)

print(f"CSV file '{output_csv}' created with {len(rows)} rows.")
