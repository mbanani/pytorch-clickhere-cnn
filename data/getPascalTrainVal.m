PASCAL3D_ROOT = '/z/home/mbanani/datasets/pascal3d';
addpath(fullfile(PASCAL3D_ROOT, 'PASCAL', 'VOCdevkit', 'VOCcode'));

% Run VOC code to extract image IDs
VOCinit;
trainImgIds = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
valImgIds = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');

% Save IDs to file
trainIdsFile = fopen('trainImgIds.txt', 'w');
for i=1:numel(trainImgIds)
    fprintf(trainIdsFile, '%s\n', trainImgIds{i});
end
valIdsFile = fopen('valImgIds.txt', 'w');
for i=1:numel(valImgIds)
    fprintf(valIdsFile, '%s\n', valImgIds{i});
end
