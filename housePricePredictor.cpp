#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>
#include <string>
#include <memory>

using namespace std;

// ------------------- CSV Reader -------------------
vector<vector<float>> read_csv(const string& filename) {
    vector<vector<float>> data;
    ifstream file(filename);
    string line;
    
    // Skip header
    getline(file, line);
    
    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string cell;
        
        while (getline(ss, cell, ',')) {
            row.push_back(stof(cell));
        }
        
        if (row.size() == 5) { // 4 features + 1 target
            data.push_back(row);
        }
    }
    return data;
}

// ------------------- Activation Functions -------------------
struct ActivationFunction {
    virtual float activate(float x) const = 0;
    virtual float derivative(float x) const = 0;
    virtual ~ActivationFunction() = default;
};

struct ReLU : ActivationFunction {
    float activate(float x) const override { return max(0.0f, x); }
    float derivative(float x) const override { return x > 0 ? 1.0f : 0.0f; }
};

struct Identity : ActivationFunction {
    float activate(float x) const override { return x; }
    float derivative(float x) const override { return 1.0f; }
};

// ------------------- Normalization -------------------
struct Normalizer {
    vector<float> mins, maxs;
    
    explicit Normalizer(const vector<vector<float>>& data) {
        if (data.empty()) return;
        
        size_t D = data[0].size();
        mins.assign(D, numeric_limits<float>::max());
        maxs.assign(D, numeric_limits<float>::min());
        
        for (const auto& row : data) {
            for (size_t i = 0; i < D; ++i) {
                mins[i] = min(mins[i], row[i]);
                maxs[i] = max(maxs[i], row[i]);
            }
        }
    }
    
    vector<float> normalize(const vector<float>& v) const {
        vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            float range = maxs[i] - mins[i];
            out[i] = (range == 0) ? 0.0f : (v[i] - mins[i]) / range;
        }
        return out;
    }
    
    vector<float> denormalize(const vector<float>& v) const {
        vector<float> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            out[i] = v[i] * (maxs[i] - mins[i]) + mins[i];
        }
        return out;
    }
};

// ------------------- Neuron and Layer -------------------
struct Neuron {
    vector<float> weights;
    float bias;
    float pre_activation;
    shared_ptr<ActivationFunction> act;
    
    Neuron(size_t inputs, shared_ptr<ActivationFunction> a) : act(a) {
        random_device rd;
        mt19937 gen(rd());
        float stddev = sqrt(2.0f / inputs); // He initialization
        normal_distribution<float> dist(0.0f, stddev);
        
        weights.resize(inputs);
        for (auto& w : weights) w = dist(gen);
        bias = dist(gen);
    }
    
    float forward(const vector<float>& in) {
        pre_activation = inner_product(in.begin(), in.end(), weights.begin(), bias);
        return act->activate(pre_activation);
    }
    
    void update(const vector<float>& inputs, float delta, float lr) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr * delta * inputs[i];
        }
        bias -= lr * delta;
    }
};

struct Layer {
    vector<Neuron> neurons;
    vector<float> last_input;
    
    Layer(size_t num_neurons, size_t inputs, shared_ptr<ActivationFunction> a) {
        for (size_t i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(inputs, a);
        }
    }
    
    vector<float> forward(const vector<float>& in) {
        last_input = in;
        vector<float> out;
        for (auto& n : neurons) {
            out.push_back(n.forward(in));
        }
        return out;
    }
};

// ------------------- Neural Network -------------------
class Network {
    vector<Layer> layers;
    Normalizer input_norm, output_norm;
    
public:
    Network(size_t in, const vector<size_t>& hidden, size_t out,
            const vector<vector<float>>& X,
            const vector<vector<float>>& Y)
        : input_norm(X), output_norm(Y) {
        size_t prev_size = in;
        for (auto h : hidden) {
            layers.emplace_back(h, prev_size, make_shared<ReLU>());
            prev_size = h;
        }
        layers.emplace_back(out, prev_size, make_shared<Identity>());
    }
    
    vector<float> predict(const vector<float>& in) {
        auto normalized = input_norm.normalize(in);
        for (auto& layer : layers) {
            normalized = layer.forward(normalized);
        }
        return output_norm.denormalize(normalized);
    }
    
    void train(const vector<vector<float>>& X, const vector<vector<float>>& Y,
               int epochs = 10000, float lr = 0.01f, float momentum = 0.9f) {
        vector<vector<vector<float>>> weight_vel(layers.size());
        vector<vector<float>> bias_vel(layers.size());
        
        // Initialize velocity buffers
        for (size_t l = 0; l < layers.size(); ++l) {
            weight_vel[l].resize(layers[l].neurons.size());
            bias_vel[l].resize(layers[l].neurons.size());
            for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
                weight_vel[l][n].resize(layers[l].neurons[n].weights.size(), 0.0f);
                bias_vel[l][n] = 0.0f;
            }
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0;
            for (size_t i = 0; i < X.size(); ++i) {
                auto x_norm = input_norm.normalize(X[i]);
                auto y_norm = output_norm.normalize(Y[i]);
                
                // Forward pass
                vector<vector<float>> activations = {x_norm};
                for (auto& layer : layers) {
                    activations.push_back(layer.forward(activations.back()));
                }
                
                // Calculate loss
                auto& output = activations.back();
                vector<float> errors(output.size());
                for (size_t j = 0; j < output.size(); ++j) {
                    errors[j] = output[j] - y_norm[j];
                    total_loss += errors[j] * errors[j];
                }
                
                // Backward pass
                for (int l = layers.size()-1; l >= 0; --l) {
                    auto& layer = layers[l];
                    vector<float> grad_prev(layer.last_input.size(), 0.0f);
                    
                    for (size_t n = 0; n < layer.neurons.size(); ++n) {
                        auto& neuron = layer.neurons[n];
                        float delta = errors[n] * neuron.act->derivative(neuron.pre_activation);
                        
                        // Update weights with momentum
                        for (size_t w = 0; w < neuron.weights.size(); ++w) {
                            weight_vel[l][n][w] = momentum * weight_vel[l][n][w] + lr * delta * layer.last_input[w];
                            neuron.weights[w] -= weight_vel[l][n][w];
                            grad_prev[w] += neuron.weights[w] * delta;
                        }
                        
                        // Update bias with momentum
                        bias_vel[l][n] = momentum * bias_vel[l][n] + lr * delta;
                        neuron.bias -= bias_vel[l][n];
                    }
                    errors = std::move(grad_prev); // Corrected with std::
                }
            }
            
            if (epoch % 1000 == 0) {
                cout << "Epoch " << epoch << " Loss: " << total_loss/X.size() << endl;
            }
        }
    }
};

// ------------------- Main Program -------------------
int main() {
    auto csv_data = read_csv("housePrice.csv");
    if (csv_data.empty()) {
        cerr << "Error: No data loaded. Check CSV file path and format!" << endl;
        return 1;
    }
    
    vector<vector<float>> X;
    vector<vector<float>> Y;
    for (const auto& row : csv_data) {
        X.push_back({row[0], row[1], row[2], row[3]}); // size, bedrooms, zip, wealth
        Y.push_back({row[4]}); // price
    }
    
    Network net(4, {8, 4}, 1, X, Y);
    net.train(X, Y, 10000, 0.001);
    
    vector<float> sample = {2400, 3, 94025, 160};
    cout << "\nPredicted Price: $" << net.predict(sample)[0] << endl;
    
    return 0;
}/*

ANN structures:

Inputs:                  Hidden Layer:        Output Layer:
+---------+              +-----+              +------+
| size    |--\           |     |              |      |
| x1      |---\          |     |---\          |      |
+---------+    \         |  o  |    \         |      |
+---------+     >--------|     |-----|------->|  y   | (price)
| bedrooms|---/   \      |     |    /         |      |
| x2      |--/     \     +-----+   /          +------+
+---------+         \             /
+---------+          \           /
| zip code|-----------\         /
| x3      |------------\       /
+---------+             \     /
+---------+              \   /
| wealth  |---------------\ /
| x4      |----------------/
+---------+
[4 input neurons] [3 hidden neurons]  [1 output neuron]
Legend:
x1 = size (sq ft)
x2 = number of bedrooms
x3 = zip code
x4 = wealth (k$)
y  = predicted price

- Each input is connected to all hidden neurons (fully connected).
- The hidden neurons are connected to the single output neuron.

data:
size (sq ft),bedrooms,zip code,wealth (k$),price (y, $)
1483,2,95051,100,712591
1272,2,94110,10,82800
1342,1,94016,137,1041220
1995,2,94121,88,698126
2721,3,94301,205,1216632
2603,3,94301,147,1001594
1690,2,94016,52,453367
2446,4,94301,187,1173380
1940,3,95014,172,1088142
1939,3,94087,85,499139
1368,2,95051,41,306895
1882,3,94016,71,569286
1864,3,95014,89,559827
1580,1,94306,100,732357
1871,3,94087,135,768286
1802,2,94087,113,689599
1956,3,94121,138,1052849
1486,4,94121,123,737723
1432,2,95051,73,654930
1487,1,95051,48,359817
1374,2,94306,56,390823
2361,4,94025,125,785965
1398,2,95051,22,186516
2135,3,94016,126,1173695
2289,2,94087,182,1420785
2234,3,94087,183,1315329
2371,3,94025,204,1408034
1188,2,94110,43,332032
1237,3,95051,93,637360
2392,5,94022,255,1211786
1568,2,95014,199,984732
1863,3,94087,187,992673
1728,2,95014,206,1249329
3000,5,94022,272,1836873
1015,2,94110,123,981801
2284,2,94025,151,872070
3134,5,94022,247,1613494
2364,4,94025,163,1008352
1643,4,95014,178,994801
1537,2,94110,124,1500693
2344,4,95014,162,1411446
2667,3,94301,226,1266411
1284,2,94016,73,476871
2277,3,95014,164,1441148
1397,3,94016,93,554532
1507,2,94016,94,749884
1357,1,95051,51,369257
1102,1,94110,10,71471
1563,2,94306,109,932318
2153,4,95014,135,915422
3102,5,94022,307,2283419
1779,4,94087,125,756455
1434,3,95051,116,948100
2369,3,94301,189,1026411
1391,4,94121,178,823758
1438,2,95051,55,405139
1513,3,94016,112,873190
969,2,95051,64,348405
1755,2,94016,61,484614
2360,3,94087,144,962540
1281,2,94306,70,386172
1223,2,94110,34,310026
980,1,95051,86,423740
1739,1,94016,84,834473
1972,2,94306,87,951082
1042,2,94016,103,480986
1919,3,94087,152,1036680
1850,2,95014,152,860733
2004,4,94087,120,832081
1397,2,94306,23,167794
1756,2,94025,204,1042304
2672,5,94022,225,1103078
1126,3,95014,170,694519
2915,4,94301,230,1489267
1919,3,94121,106,740516
1411,1,94306,75,566511
2696,5,94022,200,1295180
1862,2,94025,165,821068
1898,3,94087,110,786478
1972,1,94110,61,898003
1490,2,95051,104,925521
2397,5,94301,188,1226272
1566,2,94110,10,98835
1694,3,95014,143,909166
1599,4,94087,182,1050766
2811,6,94022,287,1576258
1580,2,94306,49,403634
3089,5,94022,268,1936997
2359,3,94301,215,1084644
1955,2,94306,97,996757
1847,2,94306,86,797723
2327,4,94301,203,1166652
2192,2,94306,88,866350
1907,4,94121,100,769025
1741,2,94016,107,1065722
2445,2,94025,174,1204883
1979,3,94121,100,731289
2576,2,94025,143,930362
2803,4,94301,194,1290176
2366,3,94121,99,824570
*/
