% Lire le fichier JSON
jsonFileName = 'coco_labels.json';
fid = fopen(jsonFileName, 'r');
if fid == -1
    error('Impossible d''ouvrir le fichier JSON');
end

raw = fread(fid, inf);
str = char(raw');
fclose(fid);

% Décoder le JSON en une structure MATLAB
jsonData = jsondecode(str);

% Vérifier si le champ "names" existe
if ~isfield(jsonData, 'names')
    error('Le champ "names" est introuvable dans le JSON');
end

% Initialiser le dictionnaire avec des clés de type 'char'
namesMap = containers.Map('KeyType', 'char', 'ValueType', 'char');

% Parcourir les champs de "names" et les ajouter au dictionnaire
nameFields = fieldnames(jsonData.names);
for i = 1:length(nameFields)
    % Récupérer la clé avec 'x' et supprimer ce préfixe
    key = nameFields{i};  
    keyWithoutX = key(2:end);  % Supprimer le 'x' du début de la clé
    value = jsonData.names.(key);
    
    % Ajouter la clé-valeur au dictionnaire
    namesMap(keyWithoutX) = value;
end

% Étape 5 : Vérifier que le dictionnaire est correctement rempli
disp('Liste des clés disponibles dans le dictionnaire :');
disp(keys(namesMap));  % Afficher toutes les clés présentes dans le dictionnaire

% Test d'accès pour s'assurer que les valeurs sont présentes
disp('Test d''accès aux valeurs :');
try
    disp(['Clé "0": ', namesMap('0')]);   % Affiche 'person'
    disp(['Clé "1": ', namesMap('1')]);   % Affiche 'bicycle'
    disp(['Clé "2": ', namesMap('2')]);   % Affiche 'car'
    disp(['Clé "4" :', namesMap('4')]);
catch ME
    disp(['Erreur lors de l''accès aux clés : ', ME.message]);
end


