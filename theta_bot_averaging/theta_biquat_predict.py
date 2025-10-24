import numpy as np
import pandas as pd
from mpmath import jtheta, mp

# Nastavení vyšší přesnosti pro výpočty Jacobiho theta funkcí
mp.dps = 50

# === Parametry skriptu ===
input_csv_path = 'eval_h_BTCUSDT_1HCSV.csv'         # cesta k vstupnímu CSV souboru s časovou řadou (očekáván sloupec 'close')
output_csv_path = 'biquat_predictions_BTCUSD.csv'  # cesta k výstupnímu CSV souboru
variant = 'A'   # zvol variantu predikčního modelu: 'A' = přímá extrapolace, 'B' = Kalmanův filtr
q = 0.5         # parametr q pro Jacobiho theta funkce (0 < q < 1)

# Parametry Kalmanova filtru (pro variantu B)
process_noise_var = 1e-3      # rozptyl procesního šumu (dynamika stavů)
measurement_noise_var = 1.0   # rozptyl šumu měření (odchylka pozorování ceny)

def generate_theta_basis(time_points, q):
    """
    Vygeneruje hodnoty Jacobiho theta funkcí θ1 až θ4 pro dané časové body.
    Návrat: numpy pole tvaru (len(time_points), 4), kde každý sloupec odpovídá jedné theta funkci.
    """
    n_funcs = [1, 2, 3, 4]
    N = len(time_points)
    basis_matrix = np.zeros((N, len(n_funcs)), dtype=float)
    # Pro každý theta index n spočítat hodnoty ve všech časech
    for j, n in enumerate(n_funcs):
        for i, t in enumerate(time_points):
            # jtheta(n, z, q) vrací hodnotu Jacobiho theta funkce pro daný n, argument z a nome q
            val = jtheta(n, t, q)
            basis_matrix[i, j] = float(val)
    return basis_matrix

def gram_schmidt_orthonormal(vecs):
    """
    Provede Gram-Schmidt ortonormalizaci seznamu vektorů.
    Vstup: seznam 1D numpy poli (každé délky N).
    Výstup: seznam 1D numpy poli, které jsou navzájem ortonormální.
    """
    ortho = []
    for v in vecs:
        u = v.astype(float).copy()
        # Odečtení projekcí na již ortonormalizované osy
        for q_vec in ortho:
            proj = np.dot(q_vec, u)
            u = u - proj * q_vec
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            # pokud vyšla téměř nulová velikost (lineární závislost), přeskočíme
            continue
        u = u / norm
        ortho.append(u)
    return ortho

def direct_extrapolation_prediction(close_values, basis_matrix):
    """
    Varianta A: Přímá extrapolace složek a složení predikované hodnoty.
    Vrací seznamy: predictions, actuals, deltas, correct_flags a index prvního predikovaného bodu.
    """
    N = len(close_values)
    M = basis_matrix.shape[1]
    start_idx = M - 1  # začínáme predikovat až po získání M bodů (aby šlo spočítat koeficienty)
    predictions = []
    actuals = []
    correct_flags = []
    for i in range(start_idx, N - 1):
        # Fit (least squares) na data 0..i pomocí bázových funkcí
        y_segment = close_values[:i+1]
        Phi = basis_matrix[:i+1, :]
        c, *_ = np.linalg.lstsq(Phi, y_segment, rcond=None)
        c = c.reshape(-1)
        # Predikce hodnoty v čase i+1 jako lineární kombinace bázových složek
        basis_next = basis_matrix[i+1, :]
        y_pred = float(np.dot(basis_next, c))
        y_true = close_values[i+1]
        # Výpočet správnosti predikce směru ('correct_pred') mezi časem i a i+1
        prev_value = close_values[i]
        pred_dir = np.sign(y_pred - prev_value)
        true_dir = np.sign(y_true - prev_value)
        if pred_dir == 0 and true_dir == 0:
            correct = 1
        elif pred_dir * true_dir > 0:
            correct = 1
        else:
            correct = 0
        predictions.append(y_pred)
        actuals.append(y_true)
        correct_flags.append(int(correct))
    deltas = [pred - act for pred, act in zip(predictions, actuals)]
    return predictions, actuals, deltas, correct_flags, start_idx

def kalman_filter_prediction(close_values, ortho_basis_matrix):
    """
    Varianta B: Kalmanův filtr pro sledování dynamiky složek.
    Využívá ortonormalizované báze jako pozorovatelné komponenty.
    Vrací seznamy: predictions, actuals, deltas, correct_flags.
    """
    N = len(close_values)
    M = ortho_basis_matrix.shape[1]
    # Počáteční stav (koeficienty složek) a jeho kovariance
    x_est = np.zeros(M)
    P_est = np.eye(M) * 1e6  # velká počáteční nejistota
    Q = process_noise_var * np.eye(M)  # kovarianční matice procesního šumu
    R = measurement_noise_var         # rozptyl měřicího šumu (skalár)
    predictions = []
    actuals = []
    correct_flags = []
    # Iterativní filtrování a predikce
    for t in range(N - 1):
        # Pozorovací vektor H_t (hodnoty bázových funkcí v čase t)
        H_t = ortho_basis_matrix[t, :]
        y_t = close_values[t]
        # **Krok aktualizace**: výpočet Kalmanova zisku K
        HPH = float(H_t @ P_est @ H_t.T)
        K = (P_est @ H_t.T) / (HPH + R)
        K = K.reshape(-1)
        # Aktualizace odhadu stavu pomocí pozorování y_t
        y_pred_t = float(H_t @ x_est)
        innovation = y_t - y_pred_t
        x_est = x_est + K * innovation
        P_est = P_est - np.outer(K, H_t) @ P_est
        # **Krok predikce**: predikce stavu a výstupu do času t+1
        x_pred = x_est.copy()    # stavový model: x_{t+1} = x_t (+ šum)
        P_pred = P_est + Q
        H_next = ortho_basis_matrix[t+1, :]
        y_pred_next = float(H_next @ x_pred)
        y_true_next = close_values[t+1]
        # Vyhodnocení správnosti predikce směru mezi t a t+1
        prev_value = close_values[t]
        pred_dir = np.sign(y_pred_next - prev_value)
        true_dir = np.sign(y_true_next - prev_value)
        if pred_dir == 0 and true_dir == 0:
            correct = 1
        elif pred_dir * true_dir > 0:
            correct = 1
        else:
            correct = 0
        predictions.append(y_pred_next)
        actuals.append(y_true_next)
        correct_flags.append(int(correct))
        # Posun stavu do dalšího kroku
        x_est = x_pred
        P_est = P_pred
    deltas = [pred - act for pred, act in zip(predictions, actuals)]
    return predictions, actuals, deltas, correct_flags

# === 1. Načtení dat ===
try:
    df = pd.read_csv(input_csv_path)
except Exception as e:
    raise SystemExit(f"Chyba při načítání CSV: {e}")
# Předpoklad: sloupec s cenou má název 'close'
if 'close' in df.columns:
    close_values = df['close'].to_numpy(dtype=float)
else:
    # pokud není, použij první číselný sloupec v datech
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            close_values = df[col].to_numpy(dtype=float)
            break
    else:
        raise ValueError("Vstupní CSV neobsahuje sloupec 'close' ani jiný číselný sloupec pro ceny.")
N = len(close_values)
time_points = np.arange(N)

# === 2. Vygenerování bází Jacobiho theta funkcí (θ1..θ4) ===
basis_matrix = generate_theta_basis(time_points, mp.mpf(q))

# === 3. Gram-Schmidt ortonormalizace těchto theta složek ===
basis_vectors = [basis_matrix[:, j] for j in range(basis_matrix.shape[1])]
ortho_vectors = gram_schmidt_orthonormal(basis_vectors)
ortho_matrix = np.column_stack(ortho_vectors) if ortho_vectors else np.empty((N, 0))

# === 4. Predikce pomocí zvolené varianty modelu ===
if variant.upper() == 'A':
    preds, acts, deltas, corrects, start_idx = direct_extrapolation_prediction(close_values, basis_matrix)
elif variant.upper() == 'B':
    preds, acts, deltas, corrects = kalman_filter_prediction(close_values, ortho_matrix)
    start_idx = 0
else:
    raise ValueError("Neznámá varianta. Nastavte variant = 'A' nebo 'B'.")

# === 5. Uložení výsledků a výpočet metrik ===
# Sestavení DataFrame s výsledky
out_df = pd.DataFrame({
    'predicted_price': preds,
    'actual_future_price': acts,
    'delta': deltas,
    'correct_pred': corrects
})
# 'correct_pred': 1 = správně predikovaný směr (up/down) proti předchozí hodnotě, 0 = nesprávně
out_df.to_csv(output_csv_path, index=False)

# Výpočet přehledových metrik
pred_arr = np.array(preds)
act_arr = np.array(acts)
if len(pred_arr) > 1:
    corr_pred_true = np.corrcoef(pred_arr, act_arr)[0, 1]
else:
    corr_pred_true = np.nan
hit_rate_pred = np.mean(corrects) if len(corrects) > 0 else np.nan
mae = np.mean(np.abs(pred_arr - act_arr)) if len(pred_arr) > 0 else np.nan

# Výpis metrik výkonu modelu
print(f"corr_pred_true: {corr_pred_true:.3f}")
print(f"hit_rate_pred: {hit_rate_pred:.3f}")
print(f"MAE: {mae:.3f}")

