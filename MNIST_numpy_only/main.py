from NeuralNetwork import NeuralNetwork

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.train_full()
n.test()