baseDir = "/Users/nellynguyen/Documents/INSA/4A/TIP/ms-coco/images";
trainDir = fullfile(baseDir, "train");
testDir = fullfile(baseDir, "test");

% Créer les imageDatastores
imdsTrain = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.png', '.jpeg'});

imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.png', '.jpeg'});

% Vérifier le chargement
disp(['Nombre d''images d''entraînement : ' num2str(numel(imdsTrain.Files))]);
disp(['Nombre d''images de test : ' num2str(numel(imdsTest.Files))]);

% Visualiser une image pour vérifier
figure;
img = readimage(imdsTrain, 1);
imshow(img);
title('Première image d''entraînement');


inputSize = [224 224 3];
