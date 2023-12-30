% Load the image
I = imread('parrot.png');

% Flatten the image matrix to an array
I_flat = I(:);
total_pixels = numel(I_flat);

% Count occurrences of each unique value
% and calculate the probability for each one
unique_values = double(unique(I_flat));
N = numel(unique_values);
probabilities = zeros(N,1);
for k = 1:N
    probabilities(k) = sum(I_flat==unique_values(k))/total_pixels;
end
value_probability_matrix = [unique_values, probabilities];

% Huffman Dictionary
huffmanDict = huffmandict(unique_values, probabilities);

% Entropy
H = -sum(probabilities .* log2(probabilities + eps));

% Avg Length of Huffman Code
avgLength = 0;
for i = 1:length(huffmanDict)
    avgLength = avgLength + probabilities(i) * length(huffmanDict{i,2});
end

%Efficiency
efficiency = (H / avgLength) * 100;

disp('First order stats');
disp(['Entropy: ', num2str(H), ' bits/symbol']);
disp(['Average Code Length: ', num2str(avgLength), ' bits/symbol']);
disp(['Huffman Coding Efficiency: ', num2str(efficiency), '%']);

% Second Order Extension

% Create pairs of adjacent pixels (horizontally) as string
I_reshaped = reshape(I_flat, 2, []);
I_pairs = transpose(I_reshaped);
I_pairs_str = num2str(I_pairs);

% Find unique pairs and their probabilities
[unique_pairs, ~, idx] = unique(I_pairs_str, 'rows');
probabilities_second_order = accumarray(idx, 1) / size(I_pairs, 1);
% Convert unique pairs back to numeric form if necessary
unique_pairs_numeric = str2num(unique_pairs);
value_probability_matrix_second_order = [unique_pairs_numeric, probabilities_second_order];


% Huffman Dictionary for pairs
symbols_pairs = 1:size(unique_pairs, 1);
huffmanDict_second_order = huffmandict(symbols_pairs, probabilities_second_order);

% Entropy
H_second_order = -sum(probabilities_second_order .* log2(probabilities_second_order + eps));

% Avg Length of Huffman Code
avgLength_second_order = 0;
for i = 1:length(huffmanDict_second_order)
    avgLength_second_order = avgLength_second_order + probabilities_second_order(i) * length(huffmanDict_second_order{i,2});
end

% Efficiency
efficiency_second_order = (H_second_order / avgLength_second_order) * 100;


disp('Second order stats');
disp(['Entropy: ', num2str(H_second_order), ' bits/pair']);
disp(['Average Code Length: ', num2str(avgLength_second_order), ' bits/pair']);
disp(['Huffman Coding Efficiency: ', num2str(efficiency_second_order), '%']);


% Using the Huffman Encoding
figure;
imshow(I);
title('Original Image');

encodedImage = huffmanenco(I_flat, huffmanDict);

decodedImageFlat = huffmandeco(encodedImage, huffmanDict);
decodedImage = uint8(reshape(decodedImageFlat, size(I)));

figure;
imshow(decodedImage);
title('Decoded Image');

% Check number of bits
bI = reshape((dec2bin(I_flat, 8) - '0').', 1, []);
numBitsOriginal = numel(bI);

% Count the number of bits in the encoded image
numBitsEncoded = length(encodedImage);

% Display the results
disp(['Number of bits in the original image: ', num2str(numBitsOriginal)]);
disp(['Number of bits in the encoded image: ', num2str(numBitsEncoded)]);
disp(['Compression Ratio: ', num2str(numBitsEncoded / numBitsOriginal)]);

% Calculate the probability of correct communication
y = binary_symmetric_channel(encodedImage);
num_correct_bits = sum(encodedImage == y);
total_bits = length(encodedImage);
estimated_p = num_correct_bits / total_bits;
disp(['Estimated probability of correct communication (p): ', num2str(estimated_p, '%.2f')]);

% Calculate the channel capacity
binary_entropy = -(estimated_p * log2(estimated_p)) - ((1 - estimated_p) * log2(1 - estimated_p));
channel_capacity = 1 - binary_entropy;
disp(['Channel Capacity: ', num2str(channel_capacity)]);

% Calculate the mutual information
probability_of_error = 1 - estimated_p;
error_entropy = -(probability_of_error * log2(probability_of_error)) + ((1 - probability_of_error) * log2(1 - probability_of_error));
mutual_information = binary_entropy - error_entropy;
disp(['Mutual Information: ', num2str(mutual_information)]);
