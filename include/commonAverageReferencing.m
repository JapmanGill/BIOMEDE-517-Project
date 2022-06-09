function [zstruct] = commonAverageReferencing(zstruct)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
neural_feature = vertcat(zstruct(:).NeuralFeature);
car = mean(neural_feature,1);
for i=1:length(zstruct)
    zstruct(i).NeuralFeature = zstruct(i).NeuralFeature - car;
end
end

