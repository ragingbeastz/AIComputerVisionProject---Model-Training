#This file intends to alter the directory structure of a the DVM-Car dataset to remove classifcation of car colours
import os
import shutil

originalDir = "C:\\Users\\Dimithri\\Downloads\\resized_DVM_v2\\resized_DVM"
savedDir = "C:\\Users\\Dimithri\\Documents\\AlteredDataset"

def alterDirectoryStructure():

    if not os.path.exists(savedDir):
        os.mkdir(savedDir)

    imageCount = 0
    progress = 0

    for make in os.listdir(originalDir):
        for model in os.listdir(os.path.join(originalDir, make)):
            for year in os.listdir(os.path.join(originalDir, make, model)):
                for item in os.listdir(os.path.join(originalDir, make, model, year)):
                    if os.path.isdir(os.path.join(originalDir, make, model, year, item)):
                        for image in os.listdir(os.path.join(originalDir, make, model, year, item)):
                            imageCount += 1
                    
                    else:
                        imageCount += 1


    print(f"Total images to copy: {imageCount}")



    for make in os.listdir(originalDir):
        os.mkdir(os.path.join(savedDir, make))
        for model in os.listdir(os.path.join(originalDir, make)):
            os.mkdir(os.path.join(savedDir, make, model))
            for year in os.listdir(os.path.join(originalDir, make, model)):
                os.mkdir(os.path.join(savedDir, make, model, year))

                for item in os.listdir(os.path.join(originalDir, make, model, year)):

                    if os.path.isdir(os.path.join(originalDir, make, model, year, item)):
                        for image in os.listdir(os.path.join(originalDir, make, model, year, item)):
                            originalPath = os.path.join(originalDir, make, model, year, item, image)
                            newPath = os.path.join(savedDir, make, model, year, image)
                            shutil.copyfile(originalPath, newPath)

                            progress += 1
                            print(f"Progress: {(progress/imageCount)*100}%")
                    
                    else:
                        originalPath = os.path.join(originalDir, make, model, year, item)
                        newPath = os.path.join(savedDir, make, model, year, item)
                        shutil.copyfile(originalPath, newPath)

                        progress += 1
                        print(f"Progress: {(progress/imageCount)*100}%")




for make in os.listdir(savedDir):
    
    for model in os.listdir(os.path.join(savedDir, make)):
        
        for year in os.listdir(os.path.join(savedDir, make, model)):
            alteredYear = year.replace(" ", "_")
            os.rename(os.path.join(savedDir, make, model, year), 
                      os.path.join(savedDir, make, model, alteredYear))
            
        alteredModel = model.replace(" ", "_")
        os.rename(os.path.join(savedDir, make, model), 
                  os.path.join(savedDir, make, alteredModel))
        
    alteredMake = make.replace(" ", "_")
    os.rename(os.path.join(savedDir, make), 
              os.path.join(savedDir, alteredMake))