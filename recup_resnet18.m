% See details of the architecture
net = net_2;
net.Layers

%freeze tout le réseau sauf la dernière couche qui apprend (head)
[layerName,learnableNames] = networkHead(net)
net = freezeNetwork(net,LayerNamesToIgnore=layerName);

deepNetworkDesigner(net)

