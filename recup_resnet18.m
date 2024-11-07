% Access the trained model
[net, classes]= imagePretrainedNetwork("resnet18");

% See details of the architecture
net.Layers
deepNetworkDesigner(net)

% Read the image to classify
I = imread('peppers.png');

% Adjust size of the image
sz = net.Layers(1).InputSize;
I = I(1:sz(1),1:sz(2),1:sz(3));

% Classify the image using ResNet-18
scores = predict(net, single(I));
label = scores2label(scores, classes)

% Show the image and the classification results
figure
imshow(I)
text(10,20,char(label),'Color','white')