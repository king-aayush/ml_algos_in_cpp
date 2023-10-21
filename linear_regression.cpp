#include<iostream>
#include<vector>
using namespace std;

class linear_regression {
    private:
    vector<double> weights;
    double intercept;

    public:
    linear_regression(vector<double>& initial_weights, double intercept = 0){
        this->intercept = intercept;
        int n = initial_weights.size();
        this->weights.resize(n);
        for(int i = 0; i < n; i++){
            this->weights[i] = initial_weights[i];
        }
    }

    void print_weights_and_intercept(){
        for(int i = 0; i < weights.size(); i++){
            cout << "x" << i << " : " << weights[i] << endl;
        }
        cout << "intercept : " << intercept << endl;
    }

    void train(vector<vector<double>>& X_train, vector<double>& y_train, double learning_rate, int max_epoch, double early_stopping = 0.01){

        double prev_cost = -1;
        int epoch = 0;

        while(true){
            if(epoch > max_epoch)
                break;

            double cost = calculate_cost(X_train, y_train);

            if(cost == 0){
                cout << "\nCongratulations perfectly fitting weights were found\n" << endl;
                break;
            }
            // cout << prev_cost << " " << cost << endl;
            if(prev_cost > 0 &&  (prev_cost - cost < early_stopping || cost > prev_cost))
                break;

            epoch++;
            gradient_descent(X_train, y_train, learning_rate);

            prev_cost = cost;

            cout << "Epoch: " << epoch << ", cost: " << cost << endl;
        }
        cout << "Training complete" << endl;
    }

    vector<double> predict(vector<vector<double>>& X){

        int m = X.size(); // number of data points
        int n = weights.size(); // number of parameters
        vector<double> predictions(m);

        for(int i = 0; i < m; i++){
            predictions[i] = intercept;
            for(int j = 0; j < n; j++){
                predictions[i] += weights[j] * X[i][j];
            }
        }
        return predictions;
    }

    private:
    double calculate_cost(vector<vector<double>>& x, vector<double>& y){
        // using mean squared error as cost function

        double m = x.size(); // number of data points
        int n = x[0].size(); // number of parameters
        double cost = 0;

        vector<double> predictions = this->predict(x);
        for(int i = 0; i < m; i++){
            cost += (((predictions[i] - y[i])*(predictions[i] - y[i])) / (2.0*m));
        }

        return cost;
    }

    void gradient_descent(vector<vector<double>>& x, vector<double>& y, double& learning_rate){
        // w_new = w_old - (learning_rate * partial_derivative)

        double m = x.size(); // number of data points
        int n = x[0].size(); // number of parameters

        double old_intercept = intercept;
        vector<double> old_weights(weights.begin(), weights.end());

        vector<double> diff(m);

        // updating for intercept
        double error_term = 0;
        for(int i = 0; i < m; i++){
            double prediction = old_intercept;
            for(int j = 0; j < n; j++){
                prediction += old_weights[j] * x[i][j];
            }
            diff[i] = prediction - y[i];
            error_term += diff[i];
        }

        intercept -= ((learning_rate * error_term) / m);

        // updating for weights
        for(int p = 0; p < n; p++){
            error_term = 0;
            for(int i = 0; i < m; i++){
                error_term += (diff[i] * x[i][p]);
            }
            weights[p] -= ((learning_rate * error_term) / m);
        }

    }
};
