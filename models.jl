# NECESSARY PACKAGES
using CSV, DataFrames, Statistics, Flux, Plots, Random

# Set a random seed
seed = rand(1:10000)  # Generate a random seed
Random.seed!(seed)    # Set the seed
seed = 9082
println("Using seed: ", seed)  # Print the seed being used

# Load and preprocess data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
data = CSV.File(download(url)) |> DataFrame
sunspots = data[:, 2]
mean_sunspots, std_sunspots = mean(sunspots), std(sunspots)
normalized_sunspots = (sunspots .- mean_sunspots) ./ std_sunspots

train_zs = Int(0.8 * length(normalized_sunspots))
train_data = normalized_sunspots[1:train_zs]
test_data = normalized_sunspots[train_zs+1:end]

function run_experiment(seed)
    Random.seed!(seed)

    # FFNN Model
    ffnn_model = Chain(Dense(1, 10, relu), Dense(10, 1))
    loss_ffnn(x, y) = Flux.Losses.mse(ffnn_model(x), y)
    learning_rate = 0.12
    opt_ffnn = Flux.Adam(learning_rate)

    X_train_ffnn = reshape(train_data[1:end-1], 1, :)
    y_train_ffnn = reshape(train_data[2:end], 1, :)

    Flux.train!(loss_ffnn, Flux.params(ffnn_model), [(X_train_ffnn, y_train_ffnn)], opt_ffnn)

    X_test_ffnn = reshape(test_data[1:end-1], 1, :)
    pred_ffnn = vec(ffnn_model(X_test_ffnn))

    # RNN Model
    rnn_model = Chain(Flux.RNN(1 => 10, tanh), Dense(10, 1))
    loss_rnn(x, y) = Flux.Losses.mse(rnn_model(x), y)
    learning_rate = 0.03
    opt_rnn = Flux.Adam(learning_rate)

    X_train_rnn = reshape(train_data[1:end-1], 1, :)
    y_train_rnn = reshape(train_data[2:end], 1, :)

    Flux.train!(loss_rnn, Flux.params(rnn_model), [(X_train_rnn, y_train_rnn)], opt_rnn)

    X_test_rnn = reshape(test_data[1:end-1], 1, :)
    Flux.reset!(rnn_model)
    pred_rnn = vec(rnn_model(X_test_rnn))

    # Calculate MSE
    mse_ffnn = Flux.Losses.mse(pred_ffnn, test_data[2:end])
    mse_rnn = Flux.Losses.mse(pred_rnn, test_data[2:end])

    return pred_ffnn, pred_rnn, mse_ffnn, mse_rnn
end

# Run the experiment
pred_ffnn, pred_rnn, mse_ffnn, mse_rnn = run_experiment(seed)

# Plotting
plot(normalized_sunspots[train_zs+1:end], label="Actual Data", title="Sunspot Predictions", xlabel="Time", ylabel="Normalized Sunspots")
plot!(pred_ffnn, label="FFNN Predictions")
plot!(pred_rnn, label="RNN Predictions")

savefig("sunspot_predictions.png")

println("FFNN MSE: ", mse_ffnn)
println("RNN MSE: ", mse_rnn)
