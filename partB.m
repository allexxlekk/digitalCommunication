% Load the source signal
load('source.mat');
signal = t;


% Define the parameters for quantization for the sample values
N = 3;  % Number of bits
min_value = -3.5;  % Minimum value of the signal
max_value = 3.5;  % Maximum value of the signal

% Define the parameters for quantization for the coefficients
N_coef = 8;  % Number of bits
min_value_coef = -2;  % Minimum value of the coefficient
max_value_coef = 2;  % Maximum value of the coefficient
p=5;

% Get the predictor's coefficients.
centers_coef = calculate_centers( N_coef, min_value_coef, max_value_coef);
coefficients = train_predictor(signal, p);
quantized_coefficients = zeros(size(coefficients));
for i = 1:length(coefficients)
    quantized_coefficients(i) = my_quantizer(coefficients(i), N_coef, min_value_coef, max_value_coef, centers_coef);
end

% Encode the signal
centers_signal = calculate_centers(N, min_value, max_value);
encoded_signal = encode_signal(signal, quantized_coefficients, N, min_value, max_value, centers_signal, p);
error_signal = signal - encoded_signal;

% Create a time vector (assuming sample rate is known or normalized time)
time_vector = 1:length(signal);
 
% Plot the original and prediction error
figure;
plot(time_vector, signal(time_vector), 'b', 'DisplayName', 'Original Signal');  % Plot original signal in blue
hold on;
plot(time_vector, error_signal, 'r', 'DisplayName', 'Prediction Error');
hold off; 
xlabel('Sample Number');
ylabel('Amplitude');
title('Original Signal vs. Prediction Error');
legend show;


% Plot the original and encoded signal
figure;
plot(time_vector, signal(time_vector), 'b', 'DisplayName', 'Original Signal');  % Plot original signal in blue
hold on;
plot(time_vector, encoded_signal(time_vector), 'r', 'DisplayName', 'Encoded Signal');
hold off; 
xlabel('Sample Number');
ylabel('Amplitude');
title('Original Signal vs. Encoded Signal');
legend show;

% Display quantized coefficients
disp('Quantized Predictor Coefficients:');
disp(quantized_coefficients);

function centers = calculate_centers( N, min_value, max_value)
    levels = 2^N;
    step_size = (max_value - min_value) / levels;

    centers = zeros(1, levels);
    centers(1) = max_value - step_size / 2;
    for i = 2:levels
        centers(i) = centers(i-1) - step_size;
    end
end

function quantized_sample = my_quantizer(y_n, N, min_value, max_value, centers)
    levels = 2^N;
    step_size = (max_value - min_value) / (levels - 1);
    y_n = max(min_value, min(y_n, max_value));
    sample_level = min(levels, max(1, round((max_value - y_n) / step_size) + 1));

    quantized_sample = centers(sample_level);
end

function a = train_predictor(X, p)
    N = length(X);
    R = zeros(p, p);
    r = zeros(p, 1);

    % Compute the elements of R and r
    for i = 1:p
        for j = 1:p
            sum_idx = (p+1):(N-max(i,j)+1);
            R(i, j) = sum(X(sum_idx) .* X(sum_idx+i-j)) / (N-p);
        end
        r(i) = sum(X((p+1):(N-i)).* X((p+1+i):N)) / (N-p);
    end

    a = R \ r;
end

function encoded_signal = encode_signal(signal, a, N, min_value, max_value, centers, p)
    encoded_signal = zeros(size(signal));  % Initialize the encoded signal array

    % Loop through the signal
    for n = 1:length(signal)
        if n > p
            % For samples after the initial p samples, use the predictor
            predicted_value = sum(a .* encoded_signal(n-1:-1:n-p));
        else
            % For initial samples, use the original signal value or a default handling
            predicted_value = 0;
        end

        % Calculate the prediction error
        prediction_error = signal(n) - predicted_value;
        % Quantize the prediction error
        prediction_error = my_quantizer(prediction_error, N, min_value, max_value, centers);

        % Store the quantized error in the encoded signal
        encoded_signal(n) = prediction_error + predicted_value;
    end
end



