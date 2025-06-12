from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import io
import numpy as np
import time

app = Flask(__name__)
class StandardScaler:
    def __init__(self):
        self.mean_ = []
        self.std_ = []

    def sqrt(self, x, tolerance=1e-10):
        if x == 0:
            return 0
        guess = x
        while True:
            new_guess = 0.5 * (guess + x / guess)
            if abs(new_guess - guess) < tolerance:
                return new_guess
            guess = new_guess

    def fit(self, X):
        if hasattr(X, "values"):  # Support DataFrame
            X = X.values.tolist()

        n_samples = len(X)
        n_features = len(X[0])
        self.mean_ = []
        self.std_ = []

        for j in range(n_features):
            col = [float(X[i][j]) for i in range(n_samples)]
            mean = sum(col) / n_samples
            variance = sum((x - mean) ** 2 for x in col) / n_samples
            std = self.sqrt(variance)
            if std == 0:
                std = 1.0
            self.mean_.append(mean)
            self.std_.append(std)
        return self

    def transform(self, X):
        if hasattr(X, "values"):  # Support DataFrame
            X = X.values.tolist()
        scaled = []
        for row in X:
            scaled_row = [(float(row[j]) - self.mean_[j]) / self.std_[j] for j in range(len(row))]
            scaled.append(scaled_row)
        return scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

scaler = joblib.load('static/Data/scalebp.pkl')
class MLPmanual0:
    def __init__(self, input_layer, hidden_layer, output_layer, epoch=100, alpha=0.01): 
        np.random.seed(42)
        self.hidden_layer = hidden_layer
        self.epoch = epoch
        self.alpha = alpha
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.inisialisasi()

    def inisialisasi(self):
        self.V = np.random.randn(self.input_layer, self.hidden_layer) * np.sqrt(2. / self.input_layer)
        self.Wi = np.random.randn(self.hidden_layer, self.hidden_layer) * np.sqrt(2. / self.hidden_layer)
        self.W = np.random.randn(self.hidden_layer, self.output_layer) * np.sqrt(2. / self.hidden_layer)  
        self.vb0 = np.zeros((1, self.hidden_layer)) 
        self.wib0=np.zeros((1, self.hidden_layer))
        self.wb1 = np.zeros((1, self.output_layer))  

    def relu(self, X):
        return np.maximum(0, X)
    
    def d_relu(self, X):
        return np.where(X > 0, 1, 0)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z))  
        return exp_Z / np.sum(exp_Z)

    def forward(self, X):
        """ 
        z1 = x . W1 + b1
        a1 = ReLU(z1)

        z2 = a1 . W2 + b2
        a2 = ReLU(z2)

        z3 = a2 . W3 + b3
        y_hat = Softmax(z3)
        """
        # --- Hidden Layer 1 ---
        self.z_inj = np.dot(X, self.V) + self.vb0        # z1 = x . V + vb0
        self.zj = self.relu(self.z_inj)                  # a1 = ReLU(z1)

        # --- Hidden Layer 2 ---
        self.z1_inj = np.dot(self.zj, self.Wi) + self.wib0  # z2 = a1 . Wi + wib0
        self.zj1 = self.relu(self.z1_inj)                   # a2 = ReLU(z2)

        # --- Output Layer ---
        self.y_ink = np.dot(self.zj1, self.W) + self.wb1    # z3 = a2 . W + wb1
        self.yk = self.softmax(self.y_ink)  

    def backward(self, tk, xi):
        """ 
        delta3 = y_hat - y_true

        grad_W3 = learning_rate * dot(a2.T delta3)
        grad_b3 = learning_rate * delta3

        delta2 = dot(delta3 W3.T) * d_ReLU(z2)

        grad_W2 = learning_rate * dot(a1.T delta2)
        grad_b2 = learning_rate * delta2

        delta1 = dot(delta2 W2.T) * d_ReLU(z1)

        grad_W1 = learning_rate * dot(x.T delta1)
        grad_b1 = learning_rate * delta1
        """
        tk_onehot = np.zeros((1, self.output_layer))
        tk_onehot[0, tk] = 1

        # --- Output Layer ---
        delta3 = self.yk - tk_onehot
        grad_W3 = self.alpha * np.dot(self.zj1.T, delta3)
        grad_b3 = self.alpha * delta3

        # --- Hidden Layer 2 ---
        delta2_in = np.dot(delta3, self.W.T)
        delta2 = delta2_in * self.d_relu(self.z1_inj)
        grad_W2 = self.alpha * np.dot(self.zj.T, delta2)
        grad_b2 = self.alpha * delta2

        # --- Hidden Layer 1 ---
        delta1_in = np.dot(delta2, self.Wi.T)
        delta1 = delta1_in * self.d_relu(self.z_inj)
        grad_W1 = self.alpha * np.dot(xi.T, delta1)
        grad_b1 = self.alpha * delta1

        # -----------------------------
        # Update bobot dan bias
        # -----------------------------
        """ 
        W3 = W3 - grad_W3
        b3 = b3 - grad_b3

        W2 = W2 - grad_W2
        b2 = b2 - grad_b2

        W1 = W1 - grad_W1
        b1 = b1 - grad_b1
        """
        
        self.W -= grad_W3
        self.wb1 -= grad_b3

        self.Wi -= grad_W2
        self.wib0 -= grad_b2

        self.V -= grad_W1
        self.vb0 -= grad_b1

        # -----------------------------
        # Cross-entropy loss
        # -----------------------------
        """ 
        loss = -sum(y_true * log(y_hat + epsilon))
        """
        epsilon = 1e-10
        self.total_error += -np.sum(tk_onehot * np.log(self.yk + epsilon))

    def fit(self, X, y):
        """ 
        x       : input data (fitur)
        W1,W2,W3: bobot antar layer (matriks)
        b1,b2,b3: bias untuk masing-masing layer
        z1,z2,z3: input ke neuron (hasil dot product + bias)
        a1,a2   : output dari hidden layers (setelah aktivasi ReLU)
        yk      : prediksi hasil softmax (probabilitas tiap kelas)
        y/tk    : label asli dalam one-hot encoding
        delta   : selisih (error) dari prediksi dan label
        grad_W  : turunan bobot (gradien) untuk pembaruan bobot
        grad_b  : turunan bias (gradien)
        learning_rate: kecepatan belajar model (Alpha)
        epsilon : nilai kecil untuk mencegah log(0)
        """
        for epoch in range(self.epoch):
            self.total_error = 0
            start_time_epoch = time.time()

            for i in range(len(X)):
                xi = X[i].reshape(1, -1)
                tk = y[i]
                self.forward(xi)
                self.backward(tk, xi)

            epoch_time = time.time() - start_time_epoch
            avg_error = self.total_error / len(X)
            if epoch % 10 == 0:
                print(f"epoch: {epoch:<6} error: {avg_error:<15.6f} waktu: {epoch_time:<12.4f}")

    def predict(self, X):
        z_inj = self.vb0 + np.dot(X, self.V)
        zj = self.relu(z_inj)

        z1_inj = self.wib0 + np.dot(zj, self.Wi)
        zj1 = self.relu(z1_inj)

        y_ink = self.wb1 + np.dot(zj1, self.W)
        yk = self.softmax(y_ink)
        return np.argmax(yk, axis=1)
model = joblib.load('static/Data/bp.pkl')

predicted_csv = None

label_mapping = {
    0: 'Normal',
    1: 'Suspect',
    2: 'Pathological'
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predicted_csv

    if 'file' not in request.files:
        return "No File Uploaded"

    file = request.files['file']
    if file.filename == '':
        return "Empty"
    try:
        df = pd.read_csv(file)

        # Daftar fitur yang wajib ada
        required_columns = [
            "prolongued_decelerations",
            "abnormal_short_term_variability",
            "percentage_of_time_with_abnormal_long_term_variability",
            "accelerations",
            "histogram_mode",
            "histogram_mean",
            "mean_value_of_long_term_variability",
            "histogram_variance",
            "histogram_median",
            "uterine_contractions",
            "baseline value",
            "histogram_tendency",
            "severe_decelerations"
        ]

        # Drop kolom target jika ada
        if 'fetal_health' in df.columns:
            df = df.drop(['fetal_health'], axis=1)

        # Cek apakah semua fitur ada
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return render_template('index.html', error=f"File tidak memiliki kolom berikut: {', '.join(missing_cols)}")

        # Urutkan kolom sesuai urutan yang diminta model
        df = df[required_columns]

        # Transformasi dan prediksi
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        predictions_labels = [label_mapping.get(p) for p in predictions]

        # Tambahkan hasil ke dataframe
        df['Detection_fetal_health'] = predictions_labels

        # Simpan ke CSV dalam memori
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        predicted_csv = csv_buffer.getvalue()

        return render_template(
            'result.html',
            table=df.to_html(classes="table table-bordered", index=False)
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=f"Error processing file: {e}")

@app.route('/download')
def download():
    global predicted_csv
    if not predicted_csv:
        return "No File Downloaded"

    return send_file(
        io.BytesIO(predicted_csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prediction.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
