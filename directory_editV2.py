import os
import shutil


originalDir = "C:\\Users\\Dimithri\\Documents\\AlteredDataset"
savedDir = "C:\\Users\\Dimithri\\Documents\\MiniDatasetV4"

if not os.path.exists(savedDir):
    os.mkdir(savedDir)

previousCount = 1451890
imageCount = 0
maxImageCount = 20
progress = 0


for make in os.listdir(originalDir):
    os.mkdir(os.path.join(savedDir, make))
    for model in os.listdir(os.path.join(originalDir, make)):
        os.mkdir(os.path.join(savedDir, make, model))
        for year in os.listdir(os.path.join(originalDir, make, model)):
            os.mkdir(os.path.join(savedDir, make, model, year))
            currentImageCount = 0
            for item in os.listdir(os.path.join(originalDir, make, model, year)):
                if currentImageCount <= maxImageCount:
                    originalPath = os.path.join(originalDir, make, model, year, item)
                    newPath = os.path.join(savedDir, make, model, year, item)
                    shutil.copyfile(originalPath, newPath)              
                    currentImageCount += 1
                    progress += 1
                    print(f"Progress: {(progress/previousCount)*100}%")
                    
                else:
                    progress += len(os.listdir(os.path.join(originalDir, make, model, year))) - currentImageCount
                    print(f"Progress: {(progress/previousCount)*100}%")
                    break




for make in os.listdir(savedDir):
    for model in os.listdir(os.path.join(savedDir, make)):
        for year in os.listdir(os.path.join(savedDir, make, model)):
            for item in os.listdir(os.path.join(savedDir, make, model, year)):
                if os.path.isdir(os.path.join(savedDir, make, model, year, item)):
                    for image in os.listdir(os.path.join(savedDir, make, model, year, item)):
                        imageCount += 1
                
                else:
                    imageCount += 1


print(f"Total images before: {previousCount}")
print(f"Total images after: {imageCount}")