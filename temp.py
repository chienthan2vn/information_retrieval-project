import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from scipy.spatial import distance as scipy_distance
from sklearn.preprocessing import RobustScaler, normalize as sk_normalize
from sklearn.metrics import average_precision_score

# Từ vit-keras
from vit_keras import vit, utils

# --- Cấu hình ---
IMAGE_SIZE = 384 # Như được đề cập trong bài báo
# Tên mô hình ViT từ bài báo (sẽ ánh xạ sang tên của vit-keras)
# ViT-B/16, ViT-L/16, ViT-B/32, ViT-L/32
# Ví dụ: 'ViT-B16', 'ViT-L16'

# --- Hàm trợ giúp ---

def get_vit_keras_model(vit_model_name_paper):
    """Tải mô hình ViT từ vit-keras dựa trên tên trong bài báo."""
    model_name_keras = ""
    if vit_model_name_paper == 'ViT-B16':
        # vit-keras có thể có các mô hình như vit.vit_b16
        # Kiểm tra tài liệu vit-keras để biết tên chính xác và tham số
        # Ví dụ: vit.vit_b16(image_size=IMAGE_SIZE, include_top=False, pretrained=True, weights='imagenet21k+imagenet2012')
        model_func = vit.vit_b16
    elif vit_model_name_paper == 'ViT-L16':
        model_func = vit.vit_l16
    elif vit_model_name_paper == 'ViT-B32':
        model_func = vit.vit_b32
    elif vit_model_name_paper == 'ViT-L32':
        model_func = vit.vit_l32
    else:
        raise ValueError(f"Tên mô hình ViT không xác định từ bài báo: {vit_model_name_paper}")

    # include_top=False để lấy đặc trưng, không phải lớp phân loại
    # weights='imagenet21k+imagenet2012' khớp với mô tả huấn luyện trước của bài báo
    # pretrained=True có thể không cần thiết nếu weights được chỉ định rõ ràng
    # một số phiên bản vit-keras có thể không có 'pretrained' hoặc xử lý nó khác
    try:
        model = model_func(
            image_size=IMAGE_SIZE,
            include_top=False,
            weights='imagenet21k+imagenet2012'
        )
    except TypeError as e:
        print(f"Lỗi TypeError khi khởi tạo {vit_model_name_paper}: {e}. Thử không có tham số 'weights' và tải sau, hoặc không có 'pretrained'.")
        # Thử một cấu hình khác nếu cấu hình trên thất bại (ví dụ: API vit-keras cũ hơn)
        try:
            model = model_func(
                image_size=IMAGE_SIZE,
                include_top=False,
                pretrained=True # Thường tải trọng số imagenet21k hoặc imagenet1k
            )
            # Nếu bạn muốn imagenet21k+imagenet2012 một cách rõ ràng và ở trên thất bại,
            # bạn có thể cần tải trọng số thủ công hoặc kiểm tra API vit-keras.
        except Exception as e2:
            raise ValueError(f"Không thể tải mô hình {vit_model_name_paper} với vit-keras: {e2}")

    model.trainable = False # Không cần huấn luyện lại
    return model


def load_and_preprocess_image_keras(image_path):
    """Tải và tiền xử lý hình ảnh theo yêu cầu của ViT và bài báo."""
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) # Chuyển đổi sang mảng numpy, 0-255

        # Tiền xử lý theo bài báo: trừ 127.5, chia tỷ lệ bởi 255 để có giá trị [-1, 1]
        # Điều này tương đương với (pixel_value / 127.5) - 1.0
        # Hoặc (pixel_value - 127.5) / 127.5
        # Giả sử img_array là 0-255
        processed_img_array = (img_array - 127.5) / 127.5

        # Thêm chiều batch
        return tf.expand_dims(processed_img_array, axis=0)
    except Exception as e:
        print(f"Lỗi khi tải hình ảnh {image_path}: {e}")
        return None

def extract_features_keras(model, image_tensor):
    """Trích xuất đặc trưng từ mô hình ViT của Keras."""
    if image_tensor is None:
        return None
    # model.predict sẽ trả về một mảng numpy
    # Đầu ra của ViT khi include_top=False thường là (batch_size, num_patches + 1, hidden_dim)
    # Đặc trưng của token [CLS] là token đầu tiên
    model_output = model.predict(image_tensor, verbose=0)
    # Lấy đặc trưng của token [CLS] (token đầu tiên)
    # Đối với vit-keras, đầu ra khi include_top=False có thể trực tiếp là vector đặc trưng
    # hoặc là chuỗi các token. Nếu là chuỗi, token [CLS] thường là đầu tiên.
    # Kiểm tra tài liệu của vit-keras về đầu ra của include_top=False.
    # Thông thường, nó sẽ là features[:, 0, :] cho token CLS.
    # Nếu đầu ra đã là (batch_size, hidden_dim), thì không cần [:, 0, :].
    if model_output.ndim == 3: # (batch, sequence, hidden_dim)
        features = model_output[:, 0, :]
    elif model_output.ndim == 2: # (batch, hidden_dim) - có thể đã được xử lý
        features = model_output
    else:
        raise ValueError(f"Định dạng đầu ra đặc trưng không mong đợi: {model_output.shape}")

    return features.squeeze() # Loại bỏ chiều batch nếu chỉ có 1 ảnh

# --- Các hàm còn lại (apply_normalization, calculate_distance, v.v.) ---
# Các hàm này hoạt động trên các mảng NumPy và không cần thay đổi đáng kể
# so với phiên bản PyTorch, miễn là đầu vào của chúng là mảng NumPy.

def apply_normalization(features, method="l2_axis1", axis=1):
    """Áp dụng các phương pháp chuẩn hóa được mô tả trong bài báo."""
    if features is None or features.ndim == 0:
        return features
    if features.ndim == 1:
        features_2d = features.reshape(1, -1)
    else:
        features_2d = features

    if method == "l1_axis1":
        return sk_normalize(features_2d, norm='l1', axis=1).squeeze()
    elif method == "l2_axis1":
        return sk_normalize(features_2d, norm='l2', axis=1).squeeze()
    elif method == "l1_axis0":
        return sk_normalize(features_2d, norm='l1', axis=0).squeeze()
    elif method == "l2_axis0":
        return sk_normalize(features_2d, norm='l2', axis=0).squeeze()
    elif method == "robust":
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        return scaler.fit_transform(features_2d).squeeze()
    else: # none
        return features.squeeze()


def calculate_distance(feat1, feat2, metric="cosine"):
    """Tính toán khoảng cách giữa hai vector đặc trưng."""
    if feat1 is None or feat2 is None:
        return float('inf')
    feat1 = np.asarray(feat1).ravel()
    feat2 = np.asarray(feat2).ravel()

    if metric == "manhattan":
        return scipy_distance.cityblock(feat1, feat2)
    elif metric == "euclidean":
        return scipy_distance.euclidean(feat1, feat2)
    elif metric == "cosine":
        return scipy_distance.cosine(feat1, feat2)
    elif metric == "braycurtis":
        return scipy_distance.braycurtis(feat1, feat2)
    elif metric == "canberra":
        return scipy_distance.canberra(feat1, feat2)
    elif metric == "chebyshev":
        return scipy_distance.chebyshev(feat1, feat2)
    elif metric == "correlation":
        if np.all(feat1 == feat1[0]) or np.all(feat2 == feat2[0]):
            return 1.0 if not np.array_equal(feat1, feat2) else 0.0
        return scipy_distance.correlation(feat1, feat2)
    else:
        raise ValueError(f"Khoảng cách không xác định: {metric}")

def calculate_ap(ranked_db_indices, relevant_indices_set):
    precisions = []
    num_relevant_retrieved = 0
    for i, db_idx in enumerate(ranked_db_indices):
        if db_idx in relevant_indices_set:
            num_relevant_retrieved += 1
            precisions.append(num_relevant_retrieved / (i + 1))
    if not precisions:
        return 0.0
    return np.mean(precisions)

def calculate_ns_score(ranked_db_indices, relevant_indices_set, top_k=4):
    count = 0
    for i in range(min(top_k, len(ranked_db_indices))):
        if ranked_db_indices[i] in relevant_indices_set:
            count += 1
    return count

def load_dataset_paths_and_ground_truth(dataset_name, dataset_base_path):
    """
    Tải đường dẫn hình ảnh truy vấn, đường dẫn hình ảnh cơ sở dữ liệu và ground truth.
    CẦN BẠN TRIỂN KHAI cụ thể cho từng tập dữ liệu.
    """
    print(f"Đang tải tập dữ liệu: {dataset_name}. BẠN CẦN TRIỂN KHAI HÀM NÀY.")
    if dataset_name == "dummy_dataset":
        q_dir = os.path.join(dataset_base_path, dataset_name, "queries")
        db_dir = os.path.join(dataset_base_path, dataset_name, "database")
        query_image_paths = [os.path.join(q_dir, f) for f in os.listdir(q_dir) if f.endswith(('.jpg', '.png'))]
        db_image_paths = [os.path.join(db_dir, f) for f in os.listdir(db_dir) if f.endswith(('.jpg', '.png'))]
        ground_truth = {}
        if query_image_paths and db_image_paths:
            query_name_to_idx = {os.path.basename(p): i for i, p in enumerate(query_image_paths)}
            db_name_to_idx = {os.path.basename(p): i for i, p in enumerate(db_image_paths)}
            gt_file = os.path.join(dataset_base_path, dataset_name, "ground_truth.txt")
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        q_name = parts[0]
                        relevant_db_names = parts[1:]
                        if q_name in query_name_to_idx:
                            q_idx = query_name_to_idx[q_name]
                            ground_truth[q_idx] = {db_name_to_idx[name] for name in relevant_db_names if name in db_name_to_idx}
            else:
                 if query_image_paths and len(db_image_paths) >=2:
                    ground_truth[0] = {0, 1} # Giả định query 0 liên quan đến db 0 và 1
        return query_image_paths, db_image_paths, ground_truth
    else:
        return [], [], {}

# --- Hàm xử lý chính ---
def evaluate_dataset_keras(vit_model_name_paper, dataset_name, dataset_base_path):
    print(f"\n--- Đánh giá mô hình: {vit_model_name_paper} trên tập dữ liệu: {dataset_name} ---")

    # Tải mô hình ViT
    try:
        vit_model = get_vit_keras_model(vit_model_name_paper)
        print(f"Đã tải mô hình {vit_model_name_paper} thành công.")
        # vit_model.summary() # Bỏ comment để xem cấu trúc model
    except Exception as e:
        print(f"Lỗi khi tải mô hình {vit_model_name_paper}: {e}")
        return


    query_paths, db_paths, ground_truth = load_dataset_paths_and_ground_truth(dataset_name, dataset_base_path)

    if not query_paths or not db_paths:
        print(f"Không có hình ảnh truy vấn hoặc cơ sở dữ liệu cho {dataset_name}. Bỏ qua.")
        return

    print("Trích xuất đặc trưng cho hình ảnh truy vấn...")
    query_features_raw = {}
    for i, q_path in enumerate(tqdm(query_paths)):
        img_tensor = load_and_preprocess_image_keras(q_path)
        query_features_raw[i] = extract_features_keras(vit_model, img_tensor)

    print("Trích xuất đặc trưng cho hình ảnh cơ sở dữ liệu...")
    db_features_raw = {}
    for i, db_path in enumerate(tqdm(db_paths)):
        img_tensor = load_and_preprocess_image_keras(db_path)
        db_features_raw[i] = extract_features_keras(vit_model, img_tensor)

    normalization_methods = ["none", "l1_axis1", "l2_axis1", "robust"] # Bỏ Axis-0 để đơn giản
    distance_metrics = ["manhattan", "euclidean", "cosine", "braycurtis", "canberra", "chebyshev", "correlation"]
    results_table = []

    for norm_method in normalization_methods:
        current_query_features = {idx: apply_normalization(feat, norm_method) for idx, feat in query_features_raw.items()}
        current_db_features = {idx: apply_normalization(feat, norm_method) for idx, feat in db_features_raw.items()}

        for dist_metric in distance_metrics:
            all_aps = []
            all_ns_scores = []

            for q_idx, q_feat in tqdm(current_query_features.items(), desc=f"Norm:{norm_method}, Dist:{dist_metric}"):
                if q_feat is None: continue
                distances = []
                valid_db_indices = []
                for db_idx, db_feat in current_db_features.items():
                    if db_feat is None: continue
                    dist = calculate_distance(q_feat, db_feat, dist_metric)
                    distances.append(dist)
                    valid_db_indices.append(db_idx)

                if not distances: continue
                ranked_indices = np.argsort(distances)
                ranked_db_indices_actual = [valid_db_indices[i] for i in ranked_indices]

                if dataset_name in ["INRIA", "Oxford5k", "Paris6k", "dummy_dataset"]:
                    relevant_set = ground_truth.get(q_idx, set())
                    if relevant_set:
                        ap = calculate_ap(ranked_db_indices_actual, relevant_set)
                        all_aps.append(ap)
                elif dataset_name == "UKBench":
                    # Cần triển khai logic ground truth cho UKBench
                    # Ví dụ: query_object_id = ground_truth_ukbench.get(q_idx)
                    # relevant_set = {idx for idx, obj_id in ground_truth_ukbench.items()
                    #                  if obj_id == query_object_id and idx != q_idx}
                    # ns_score = calculate_ns_score(ranked_db_indices_actual, relevant_set)
                    # all_ns_scores.append(ns_score)
                    pass

            if dataset_name in ["INRIA", "Oxford5k", "Paris6k", "dummy_dataset"]:
                mAP = np.mean(all_aps) if all_aps else 0.0
                results_table.append({
                    "Model": vit_model_name_paper, "Dataset": dataset_name,
                    "Normalization": norm_method, "Distance": dist_metric,
                    "Score_Type": "mAP", "Score": mAP
                })
            elif dataset_name == "UKBench":
                avg_ns_score = np.mean(all_ns_scores) if all_ns_scores else 0.0
                results_table.append({
                    "Model": vit_model_name_paper, "Dataset": dataset_name,
                    "Normalization": norm_method, "Distance": dist_metric,
                    "Score_Type": "N-S", "Score": avg_ns_score
                })

    print(f"\n--- Bảng kết quả cuối cùng cho {vit_model_name_paper} trên {dataset_name} ---")
    for res in results_table:
        print(f"Model: {res['Model']:<8} | Dataset: {res['Dataset']:<10} | Norm: {res['Normalization']:<10} | Dist: {res['Distance']:<12} | {res['Score_Type']}: {res['Score']:.4f}")

# --- Điểm vào chính ---
if __name__ == "__main__":
    # Đảm bảo TensorFlow có thể thấy GPU nếu có
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Đã tìm thấy và cấu hình GPUs: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("Không tìm thấy GPU, chạy trên CPU.")

    DATASET_BASE_PATH = "./datasets"

    # Tạo tập dữ liệu giả để thử nghiệm
    if not os.path.exists(os.path.join(DATASET_BASE_PATH, "dummy_dataset")):
        print("Tạo tập dữ liệu giả...")
        os.makedirs(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "queries"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "database"), exist_ok=True)
        try:
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = 'red').save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "queries", "query_0.jpg"))
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = 'blue').save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "queries", "query_1.jpg"))
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = 'red').save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "database", "db_0.jpg"))
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = (200,0,0)).save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "database", "db_1.jpg"))
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = 'green').save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "database", "db_2.jpg"))
            Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = 'blue').save(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "database", "db_3.jpg"))
            with open(os.path.join(DATASET_BASE_PATH, "dummy_dataset", "ground_truth.txt"), 'w') as f:
                f.write("query_0.jpg db_0.jpg db_1.jpg\n")
                f.write("query_1.jpg db_3.jpg\n")
        except Exception as e:
            print(f"Không thể tạo hình ảnh giả: {e}")


    # Các mô hình được đề cập trong bài báo (Bảng IV và V)
    # ViT-B/16 và ViT-L/16 dường như là các mô hình chính được so sánh trong bảng V
    # và cũng được liệt kê trong Bảng I, II, III, IV với các biến thể /16 và /32.
    # Chúng ta sẽ chạy với một số biến thể.
    paper_vit_models_to_test = ['ViT-B16'] #, 'ViT-L16', 'ViT-B32'] # Thêm các mô hình khác nếu muốn
    datasets_to_evaluate = ["dummy_dataset"] #, "INRIA", "UKBench", "Paris6k", "Oxford5k"]

    for model_name in paper_vit_models_to_test:
        for dataset_name in datasets_to_evaluate:
            evaluate_dataset_keras(model_name, dataset_name, DATASET_BASE_PATH)

    print("\nHoàn thành đánh giá.")
